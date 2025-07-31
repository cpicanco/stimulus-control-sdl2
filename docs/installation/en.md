# Installation

The development environment on Windows (x86_64-win64) includes:

- Lazarus IDE trunk version (3.4+), the development interface.
- Free Pascal Compiler trunk version (3.3.1+), the compiler.
- The project repository with its dependencies on GitHub.

The following steps assume you have a good broadband connection and will take approximately 45 minutes.

# Step-by-Step Guide

## Step 1 (20 min) - fpcupdeluxe, Lazarus IDE, FPC
1. Create the folder `C:\fpcupdeluxe\`
2. Create the folder `C:\lazarus-trunk\`
3. Download the file [fpcupdeluxe-x86_64-win64.exe](https://github.com/LongDirtyAnimAlf/fpcupdeluxe/releases/download/v2.4.0e/fpcupdeluxe-x86_64-win64.exe) from the version manager [fpcupdeluxe](https://github.com/LongDirtyAnimAlf/fpcupdeluxe/releases). `fpcupdeluxe` will be used to install the IDE and the compiler.
4. Copy the file `fpcupdeluxe-x86_64-win64.exe` to the `fpcupdeluxe` folder.
5. Open the file `fpcupdeluxe-x86_64-win64.exe`.
6. Click the `Set install path` button and choose the folder `C:\lazarus-trunk\`.
7. In the `FPC version` list, select `trunk`.
8. In the `Lazarus version` list, select `trunk`.
9. Click the `Setup+` button and check the `Include help` option under `Miscellaneous settings`.
10. Click the `Install/update FPC+Lazarus` button and wait for the process to complete.

## Step 2 (4 min) - Monokai Dark Theme for Lazarus (optional)
1. Download the `Monokai.xml` file with the [Monokai theme](https://wiki.freepascal.org/UserSuppliedSchemeSettings).
2. Copy the `Monokai.xml` file to the folder `C:\lazarus-trunk\config_lazarus\userschemes`.
3. Open Lazarus.
4. Navigate to Tools > Options > Editor > Display > Colors, and select `Monokai` in the `Color Schemes` drop-down menu.
5. Go to Lazarus > Tools > Options > Editor > Display font, and set it to `Consolas`, size 9.
6. Install [metadarkstyle](https://github.com/zamtmn/metadarkstyle) package (tip: available at Online Package Manager)

## Step 3 (1 min) - Download Dependencies
The project depends on three libraries:

- zmq 4.x (https://zeromq.org/)
- Eye Link (https://www.sr-research.com/support/forum-9.html, requires registration)
- SDL2 (https://github.com/libsdl-org/SDL/releases)

1. You can download the DLLs [here](https://drive.google.com/drive/folders/1DVSJrth2xP6rerUs1RnUYDRQWJoM7YhA?usp=sharing).

## Step 4 (20 min) - Git Bash, Stimulus Control
1. Download and install Git for Windows from https://git-scm.com/download/win.
2. Open `Git Bash` (command prompt).
3. Recursively clone the repository by running the following command (tip: use `SHIFT+Insert` to paste text into the prompt):
    ```
    mkdir sc
    git clone --recursive https://github.com/cpicanco/stimulus-control-sdl2.git sc
    ```
4. Copy the `dlls` from Step 3 into the `sc` folder.
5. Open Lazarus (always using the shortcut created by `fpcupdeluxe` on your desktop).
6. Install the `Online Package Manager` package.
   1. Navigate to Package > Install/Uninstall Packages.
   2. Type `online` in the search box on the right.
   3. Click `OnlinePackageManager 1.x.x.x`.
   4. Click `Install selection`.
   5. Click `Rebuild IDE`.
7. Navigate to Project > Open Project.
8. Open the `experiment.lpi` project located in the `sc` folder.
9. Click `Yes` to install any missing packages (rgbabitmap and synaser).
10. Click the drop-down `Change Build Mode` (gear icon) to select the desired build.
11. Press F9 or click the `Run` button (green arrow) to compile.
