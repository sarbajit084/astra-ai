# Astra AI - Android Mobile Application

This directory contains the complete, native Android project for the **Astra RAG + AI Agent** mobile application.

---

## 1. Features & Capabilities

- **Native WebView Performance**: Hardware-accelerated rendering with smooth touch scrolling and responsive layout.
- **Document Upload Support**: Integrated native `WebChromeClient` file chooser supporting PDF, Word (`.docx`), plain text, markdown, CSV, and `.zip` archives.
- **Immersive Astra Styling**: Customized dark system status bar (`#0B0F17`) matching Astra's glassmorphic UI.
- **Offline & Reconnection Handling**: Automatic detection of network disconnection with retry banner and graceful reconnection.
- **Universal Production Connectivity**: Communicates with your live GoDaddy server or remote cloud backend over HTTPS without hardcoded localhost restrictions.

---

## 2. Connecting to Your Production Backend

By default, the bundled mobile app points to the production domain placeholder:
```javascript
window.ASTRA_API_BASE_URL = 'https://YOUR-DOMAIN.com';
```

### Option A: Configure Before Building
Run the bundle script with your live domain:
```bash
python bundle_assets.py https://your-godaddy-domain.com
```
Or manually edit `android/app/src/main/assets/www/config.js`:
```javascript
window.ASTRA_API_BASE_URL = 'https://your-godaddy-domain.com';
```

### Option B: Configure at Runtime
You can also set the API base URL inside the app via the browser console or `localStorage`:
```javascript
localStorage.setItem('astra_api_base_url', 'https://your-godaddy-domain.com');
```

---

## 3. Building the Release APK

### Method 1: Using the Automated Build Script (CLI)
Ensure Python and Java 17 are installed, then run:
```bash
python build_astra_apk.py
```
The output APK will be placed at:
- `Astra-Android-Release.apk` (Project Root)
- `dist/Astra-Android-Release.apk`

### Method 2: Using Android Studio
1. Open **Android Studio**.
2. Select **Open** and choose the `android/` directory (`d:/rag_agent/android`).
3. Allow Gradle to sync project dependencies.
4. Select **Build** > **Build Bundle(s) / APK(s)** > **Build APK(s)** (or **Generate Signed APK**).
5. The generated APK will be in `app/build/outputs/apk/release/app-release.apk`.

---

## 4. Installing the APK on Your Android Device

### Method A: Direct Transfer / Download
1. Copy `Astra-Android-Release.apk` to your phone via USB cable, WhatsApp, Google Drive, or email.
2. Tap the file on your device.
3. If prompted, enable **"Install unknown apps"** for your file manager or browser.
4. Tap **Install** and open **Astra**.

### Method B: Using ADB (Android Debug Bridge)
Connect your Android phone with USB Debugging enabled and run:
```bash
adb install Astra-Android-Release.apk
```

---

## 5. Project Structure

```
android/
├── app/
│   ├── build.gradle              # App build configuration (minSdk 24, targetSdk 34)
│   └── src/main/
│       ├── AndroidManifest.xml   # Permissions (INTERNET, ACCESS_NETWORK_STATE) & activity config
│       ├── java/com/astra/ai/
│       │   └── MainActivity.java # WebView, file chooser, and lifecycle handler
│       ├── assets/www/           # Production web assets and config.js
│       └── res/                  # App icons and XML configs
├── build.gradle                  # Top-level Gradle project configuration
├── settings.gradle               # Project modules definition
└── local.properties              # Local Android SDK location
```
