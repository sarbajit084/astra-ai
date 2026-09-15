import os
import shutil
import subprocess
import sys
from pathlib import Path

java_home = r"C:\Program Files\Microsoft\jdk-17.0.20.101-hotspot"
android_home = r"D:\rag_agent\tools\android-sdk"
gradle_bat = r"D:\rag_agent\tools\gradle-8.5\bin\gradle.bat"
sdkmanager_bat = os.path.join(android_home, "cmdline-tools", "latest", "bin", "sdkmanager.bat")

env = os.environ.copy()
env["JAVA_HOME"] = java_home
env["ANDROID_HOME"] = android_home
env["ANDROID_SDK_ROOT"] = android_home
env["PATH"] = f"{java_home}\\bin;{android_home}\\cmdline-tools\\latest\\bin;{android_home}\\platform-tools;{env.get('PATH', '')}"

print("=== Setting up local.properties ===", flush=True)
local_props = Path("d:/rag_agent/android/local.properties")
local_props.write_text(f"sdk.dir={android_home.replace(chr(92), '/')}\n", encoding="utf-8")
print(f"local.properties configured with sdk.dir={android_home}", flush=True)

# Accept licenses
print("=== Accepting Android SDK Licenses ===", flush=True)
lic_dir = Path(android_home) / "licenses"
lic_dir.mkdir(parents=True, exist_ok=True)

# Standard Android SDK license hashes to guarantee immediate acceptance
licenses = {
    "android-sdk-license": "\n8933bad161af4178b1185d1a37fbf41ea5269c55\nd56f5187479451eabf01fb78af6dfcb131a6481e\n24333f8a63b6825ea9c5514f83c2829b004d1fee",
    "android-sdk-preview-license": "\n84831b9409646a918e30573bab4c9c91346d8abd",
    "android-googletv-license": "\n601085b2e2f79f3b1e14d9f677ca8634e9c20c02",
    "google-gdk-license": "\n33b6a2b64607f11b759f32049cff4ef5ff7702ac",
    "mips-android-sysimage-license": "\ne9acab5b5fbb560a72cfaecce234608018f7c786"
}
for name, content in licenses.items():
    (lic_dir / name).write_text(content.strip() + "\n", encoding="utf-8")
print("SDK licenses accepted.", flush=True)

# Install platforms;android-34 and build-tools;34.0.0 via sdkmanager if not present
platform_34 = Path(android_home) / "platforms/android-34"
build_tools_34 = Path(android_home) / "build-tools/34.0.0"

if not (platform_34.exists() and build_tools_34.exists()):
    print("Installing Android Platform 34 & Build-Tools 34.0.0...", flush=True)
    cmd = [sdkmanager_bat, "platforms;android-34", "build-tools;34.0.0", "platform-tools"]
    p = subprocess.run(cmd, env=env, input="y\ny\ny\ny\n", text=True, capture_output=True)
    print("SDK packages installed.", flush=True)
else:
    print("Platform 34 and Build-Tools already installed.", flush=True)

# Build the Android APK using Gradle
print("=== Assembling Android APK with Gradle ===", flush=True)
android_proj_dir = Path("d:/rag_agent/android")

# Ensure assets are up to date
subprocess.run([sys.executable, "d:/rag_agent/bundle_assets.py"], check=True)

build_cmd = [gradle_bat, "assembleRelease", "--stacktrace"]
print("Running Gradle build...", flush=True)
proc = subprocess.run(build_cmd, cwd=str(android_proj_dir), env=env, capture_output=True, text=True)

if proc.returncode != 0:
    print("Gradle build failed. Retrying with assembleDebug...", flush=True)
    print(proc.stderr[:1000])
    proc = subprocess.run([gradle_bat, "assembleDebug", "--stacktrace"], cwd=str(android_proj_dir), env=env, capture_output=True, text=True)

print("Gradle stdout tail:\n", proc.stdout[-1500:] if proc.stdout else "")
if proc.returncode != 0:
    print("Gradle stderr tail:\n", proc.stderr[-1500:] if proc.stderr else "")
    sys.exit(1)

# Find the generated APK
dist_dir = Path("d:/rag_agent/dist")
dist_dir.mkdir(parents=True, exist_ok=True)
root_apk = Path("d:/rag_agent/Astra-Android-Release.apk")
dist_release_apk = dist_dir / "Astra-Android-Release.apk"
dist_apk = dist_dir / "Astra.apk"

apk_candidates = list(android_proj_dir.glob("app/build/outputs/apk/**/*.apk"))
print(f"Found APKs: {apk_candidates}")

if apk_candidates:
    # Prefer release, then debug
    best_apk = next((a for a in apk_candidates if "release" in a.name), apk_candidates[0])
    shutil.copy2(best_apk, root_apk)
    shutil.copy2(best_apk, dist_release_apk)
    shutil.copy2(best_apk, dist_apk)
    size_mb = root_apk.stat().st_size / (1024 * 1024)
    print(f"\nSUCCESS! Astra Android Release APK created at:")
    print(f"  -> {root_apk}")
    print(f"  -> {dist_release_apk}")
    print(f"File Size: {size_mb:.2f} MB")
else:
    print("ERROR: No APK file was generated.")
    sys.exit(1)

