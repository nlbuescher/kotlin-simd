import org.jetbrains.kotlin.gradle.plugin.mpp.*
import org.jetbrains.kotlin.gradle.utils.*
import org.jetbrains.kotlin.konan.target.*
import org.jreleaser.model.*

plugins {
	alias(libs.plugins.kotlin.multiplatform)
	alias(libs.plugins.dokka)
	alias(libs.plugins.maven.publish)
	alias(libs.plugins.jreleaser)
}

group = "dev.buescher"
version = providers
	.exec { commandLine("git", "describe", "--abbrev=0", "--tags") }
	.standardOutput.asText
	.getOrElse("v0.0.0")
	.let { previousVersion ->
		val isExactMatch = providers
			.exec {
				commandLine("git", "describe", "--tags", "--exact-match")
				isIgnoreExitValue = true
			}
			.result.map { it.exitValue == 0 }
			.getOrElse(false)

		if (isExactMatch) {
			previousVersion
		}
		else {
			val result = Regex("""^v(?<major>\d+)\.(?<minor>\d+)\.(?<patch>\d+)""")
				.find(previousVersion)
				?: error("previous version '$previousVersion' does not follow semantic versioning!")
			val (major, minor, patch) = result.groupValues.drop(1).map { it.toInt() }
			"$major.${minor + 1}.0-SNAPSHOT"
		}
	}
	.also { logger.lifecycle("project version: {}", it) }

repositories {
	mavenCentral()
}

val projectDir: Directory = layout.projectDirectory
val buildDir: Directory = layout.buildDirectory.get()

val cppSrcDir = projectDir.dir("src/nativeInterop/cinterop/simd")
val cppBuildDir = buildDir.dir("simd")

val platformManager: PlatformManager by lazy {
	val downloader = NativeCompilerDownloader(project)
		.also { it.downloadIfNeeded() }

	PlatformManager(downloader.compilerDirectory.absolutePath)
}

val simdNames = listOf("sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2")

fun registerBuildSimdTaskForTarget(konanTarget: KonanTarget): TaskProvider<Exec> {
	val targetBuildDir = cppBuildDir.dir(konanTarget.presetName)
	val presetSuffix = konanTarget.presetName.replaceFirstChar { it.uppercase() }

	val platform = platformManager.platform(konanTarget)

	val compileTasks = simdNames.map { name ->
		val objectSuffix = name.replaceFirstChar { it.uppercase() }

		tasks.register<Exec>("compile$objectSuffix$presetSuffix") {
			group = "build"

			val sourceFile = cppSrcDir.file("$name.cpp")
			val objectFile = targetBuildDir.file("$name.o")

			inputs.files(cppSrcDir.asFileTree.filter { it.name.startsWith(name) })
			outputs.file(targetBuildDir.file("$name.o"))

			commandLine(
				platform.clang.clangCXX("-std=c++20", "-O3", "-fPIC", "-m$name", "-c", "-o", "$objectFile", "$sourceFile"),
			)
		}
	}

	val compileSimd = tasks.register<Exec>("compileSimd$presetSuffix") {
		group = "build"

		compileTasks.forEach {
			dependsOn(it)
		}

		val sourceFile = cppSrcDir.file("simd.cpp")
		val objectFile = targetBuildDir.file("simd.o")

		inputs.files(cppSrcDir.asFileTree.filter { it.name.endsWith(".h") })
		inputs.file(sourceFile)
		outputs.file(objectFile)

		commandLine(platform.clang.clangCXX("-std=c++20", "-O3", "-fPIC", "-c", "-o", "$objectFile", "$sourceFile"))
	}

	val buildSimd = tasks.register<Exec>("buildSimd$presetSuffix") {
		group = "build"

		dependsOn(compileSimd)

		val objectFiles = (listOf(targetBuildDir.file("simd.o")) + simdNames.map { targetBuildDir.file("$it.o") })
			.map { it.toString() }
			.toTypedArray()
		val archiveFile = targetBuildDir.file("libsimd.a")

		compileTasks.forEach { inputs.files(it.get().outputs.files) }
		inputs.files(compileSimd.get().outputs.files)
		outputs.file(archiveFile)

		commandLine(platform.clang.llvmAr("-rcs", "$archiveFile", *objectFiles))
	}

	return buildSimd
}

kotlin {
	withSourcesJar()

	linuxX64()
	macosX64()
	mingwX64()

	targets.withType<KotlinNativeTarget> {
		compilations.named("main") {
			cinterops.create("simd") {
				tasks[interopProcessingTaskName].run {
					inputs.files(cppSrcDir.asFileTree.filter { it.name.endsWith(".h") })
				}
				includeDirs(cppSrcDir)
			}

			if (platformManager.isEnabled(konanTarget)) {
				val buildSimdTask = registerBuildSimdTaskForTarget(konanTarget)
				compileTaskProvider.configure {
					dependsOn(buildSimdTask)
				}
			}
		}

		compilerOptions {
			freeCompilerArgs = listOf("-include-binary", "${cppBuildDir.dir(konanTarget.presetName).file("libsimd.a")}")
		}
	}

	sourceSets {
		all {
			languageSettings {
				optIn("kotlinx.cinterop.ExperimentalForeignApi")
			}
		}
		commonMain.dependencies {
			implementation(kotlin("test"))
		}
	}
}

val dokkaJar by tasks.registering(Jar::class) {
	from(tasks.dokkaGeneratePublicationHtml.flatMap { it.outputDirectory })
	archiveClassifier.set("javadoc")
}

val repositoryUrl = "https://github.com/nlbuescher/kotlin-simd"
val stagingDir = buildDir.dir("staging-deploy")

publishing {
	repositories {
		maven {
			name = "staging"
			url = uri(stagingDir)
		}
	}
	publications {
		withType<MavenPublication> {
			artifact(dokkaJar)

			pom {
				name.set(rootProject.name)
				description.set("SIMD for Kotlin/Native")
				url.set(repositoryUrl)
				licenses {
					license {
						name.set("MIT")
						url.set("https://opensource.org/licenses/MIT")
					}
				}
				issueManagement {
					system.set("Github")
					url.set("$repositoryUrl/issues")
				}
				scm {
					connection.set("$repositoryUrl.git")
					url.set(repositoryUrl)
				}
				developers {
					developer {
						name.set("Nico Büscher")
						email.set("nico@buescher.dev")
					}
				}
			}
		}
	}
}

val publishTasks = HostManager.host
	.let {
		when (it.family) {
			Family.LINUX -> listOf("LinuxX64")
			Family.OSX -> listOf("MacosX64", "KotlinMultiplatform")
			Family.MINGW -> listOf("MingwX64")
			else -> error("Unknown host family '${it.family}'")
		}
	}
	.map { tasks.named("publish${it}PublicationToStagingRepository") }

tasks.register("smartPublish") {
	group = "publishing"

	publishTasks.forEach {
		dependsOn(it)
	}
}

jreleaser {
	signing {
		pgp {
			active = Active.ALWAYS
			armored = true
		}
	}
	deploy {
		maven {
			mavenCentral {
				create("central") {
					active = Active.RELEASE
					url = "https://central.sonatype.com/api/v1/publisher"
					stagingRepository(stagingDir)

					kotlin.targets.withType<KotlinNativeTarget> {
						if (platformManager.isEnabled(konanTarget)) {
							val publication = publishing.publications.getByName<MavenPublication>(name)

							artifactOverride {
								artifactId = publication.artifactId
								jar = false
								sourceJar = true
								javadocJar = true
							}
						}
					}
				}
			}
			nexus2 {
				create("snapshots") {
					active = Active.SNAPSHOT
					snapshotUrl = "https://central.sonatype.com/repository/maven-snapshots/"
					verifyPom = false
					applyMavenCentralRules = true
					snapshotSupported = true
					closeRepository = true
					releaseRepository = true
					stagingRepository(stagingDir)

					kotlin.targets.withType<KotlinNativeTarget> {
						if (platformManager.isEnabled(konanTarget)) {
							val publication = publishing.publications.getByName<MavenPublication>(name)

							artifactOverride {
								artifactId = publication.artifactId
								jar = false
								sourceJar = true
								javadocJar = true
							}
						}
					}
				}
			}
		}
	}
}
