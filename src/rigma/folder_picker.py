"""The native Explorer folder picker for a chat's workspace (Windows).

MEASURED 2026-10-05: setting the workspace was a text box and nothing else.
`frontend-v2/src/WorkspacePanel.tsx:92-99` and `App.tsx:129-142` both hand the
owner a `placeholder="D:\\Writing"` input, and the repo held no dialog API at
all — no `FolderBrowserDialog`, no `IFileDialog`, no tkinter, no
`SHBrowseForFolder` — so the only way to point a chat at a folder was to type an
absolute path from memory.

`IFileOpenDialog` with `FOS_PICKFOLDERS` is Explorer's own picker. The obvious
alternative is NOT the same dialog: this machine's
`System.Windows.Forms.FolderBrowserDialog` has no `AutoUpgradeEnabled` property
(verified by reflection), which is the tell that it is the legacy tree-style
"Browse for Folder" dialog. The shell picker is COM, so it needs a C# interop
definition and an STA thread — hence PowerShell, the tool `probe.py:126-129`
already sanctions with a timeout.

The dialog is drawn by THIS process on ITS desktop. A server started as a
service, from session 0, or over SSH has no desktop to draw on, and that is
reported as a reason rather than as an empty path: "the owner cancelled" and
"there was nobody to ask" must not look identical to the caller.
"""
from __future__ import annotations

import os
import platform
import subprocess
from typing import NamedTuple


class Picked(NamedTuple):
    """`path` when the owner chose a folder, `cancelled` when they closed the
    dialog, `reason` when no dialog could be shown at all."""

    path: str = ""
    cancelled: bool = False
    reason: str = ""


# `-STA` is not optional: the shell dialog is a COM apartment-threaded object and
# `Show` fails outright from an MTA. `-NonInteractive` only silences PowerShell's
# OWN prompts, never the dialog.
_PS_FLAGS = ["powershell", "-NoProfile", "-NonInteractive", "-STA", "-Command"]

_PS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$code = @'
using System;
using System.Runtime.InteropServices;

[ComImport, Guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")]
internal class FileOpenDialogRCW { }

[ComImport, Guid("42f85136-db7e-439c-85f1-e4075d135fc8"),
 InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
internal interface IFileDialog {
  [PreserveSig] int Show(IntPtr parent);
  void SetFileTypes(uint cFileTypes, IntPtr rgFilterSpec);
  void SetFileTypeIndex(uint iFileType);
  void GetFileTypeIndex(out uint piFileType);
  void Advise(IntPtr pfde, out uint pdwCookie);
  void Unadvise(uint dwCookie);
  void SetOptions(uint fos);
  void GetOptions(out uint pfos);
  void SetDefaultFolder(IShellItem psi);
  void SetFolder(IShellItem psi);
  void GetFolder(out IShellItem ppsi);
  void GetCurrentSelection(out IShellItem ppsi);
  void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string pszName);
  void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string pszName);
  void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string pszTitle);
  void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string pszText);
  void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string pszLabel);
  void GetResult(out IShellItem ppsi);
  void AddPlace(IShellItem psi, int fdap);
  void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string pszDefaultExtension);
  void Close(int hr);
  void SetClientGuid(ref Guid guid);
  void ClearClientData();
  void SetFilter(IntPtr pFilter);
}

[ComImport, Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe"),
 InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
internal interface IShellItem {
  void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
  void GetParent(out IShellItem ppsi);
  void GetDisplayName(uint sigdnName,
                      [MarshalAs(UnmanagedType.LPWStr)] out string ppszName);
  void GetAttributes(uint sfgaoMask, out uint psfgaoAttribs);
  void Compare(IShellItem psi, uint hint, out int piOrder);
}

public static class RigmaFolderPicker {
  [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
  private static extern void SHCreateItemFromParsingName(
      [MarshalAs(UnmanagedType.LPWStr)] string pszPath, IntPtr pbc,
      ref Guid riid, [MarshalAs(UnmanagedType.Interface)] out IShellItem ppv);

  public static string Pick(string initial) {
    var dlg = (IFileDialog)new FileOpenDialogRCW();
    // FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST: folders only,
    // and only real filesystem ones (a virtual shell folder like "This PC" has
    // no path to store).
    dlg.SetOptions(0x20u | 0x40u | 0x800u);
    if (!String.IsNullOrEmpty(initial) && System.IO.Directory.Exists(initial)) {
      Guid iid = typeof(IShellItem).GUID;
      IShellItem start;
      SHCreateItemFromParsingName(initial, IntPtr.Zero, ref iid, out start);
      if (start != null) { dlg.SetFolder(start); }
    }
    if (dlg.Show(IntPtr.Zero) != 0) { return null; }   // ERROR_CANCELLED
    IShellItem item;
    dlg.GetResult(out item);
    string path;
    item.GetDisplayName(0x80058000u, out path);        // SIGDN_FILESYSPATH
    return path;
  }
}
'@
Add-Type -TypeDefinition $code -Language CSharp
try {
  $picked = [RigmaFolderPicker]::Pick($env:RIGMA_PICK_INITIAL)
} catch {
  # A folder can still be chosen with the legacy tree dialog, so fall back
  # rather than failing the request — it is not Explorer, but it is not a dead
  # end either.
  Add-Type -AssemblyName System.Windows.Forms
  $fb = New-Object System.Windows.Forms.FolderBrowserDialog
  $fb.Description = 'Choose the folder this chat works in'
  if ($env:RIGMA_PICK_INITIAL -and (Test-Path $env:RIGMA_PICK_INITIAL)) {
    $fb.SelectedPath = $env:RIGMA_PICK_INITIAL
  }
  if ($fb.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    $picked = $fb.SelectedPath
  } else { $picked = $null }
}
if ($picked) { Write-Output ('OK:' + $picked); exit 0 }
exit 3
"""


def pick_folder(initial: str = "", *, timeout: float = 300.0) -> Picked:
    """Show the OS folder picker and return what the owner chose.

    Blocking, and for as long as a human takes: every caller must run this off
    the event loop (`asyncio.to_thread`), because the dialog is modal and the
    server holds one loop for every live stream.

    `timeout` is a real cancel path, not just a guard: `subprocess.run` kills the
    child on expiry, which closes the window.
    """
    if platform.system() != "Windows":
        return Picked(reason="the native folder picker is Windows-only")
    env = {**os.environ, "RIGMA_PICK_INITIAL": str(initial or "")}
    try:
        proc = subprocess.run([*_PS_FLAGS, _PS_SCRIPT], capture_output=True,
                              text=True, timeout=timeout, env=env)
    except FileNotFoundError:
        return Picked(reason="powershell was not found on PATH")
    except subprocess.TimeoutExpired:
        return Picked(reason=f"the folder picker was not answered within "
                             f"{int(timeout)}s")
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("OK:"):
            return Picked(path=line[3:].strip())
    if proc.returncode == 3:
        return Picked(cancelled=True)
    detail = " ".join((proc.stderr or "").split())[-300:]
    return Picked(reason=detail or
                  f"the folder picker exited with code {proc.returncode}")
