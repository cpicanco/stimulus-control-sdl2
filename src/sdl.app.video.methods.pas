{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.video.methods;

{$mode ObjFPC}{$H+}

interface

uses SDL2, SysUtils;

procedure AssignVariables(APSDLWindow : PSDL_Window;
  APSDLRenderer : PSDL_Renderer; APSDLSurface : PSDL_Surface);
function MonitorFromWindow : TSDL_Rect;
procedure InvalidateWindow;
procedure RaiseWindow;
function WindowDeviceContextHandle : THandle;
function WindowHandle : THandle;
function WindowBoundsRect : TSDL_Rect;

var
  PSDLRenderer : PSDL_Renderer;
  PSDLSurface  : PSDL_Surface;
  PSDLWindow   : PSDL_Window;

implementation

procedure AssignVariables(APSDLWindow: PSDL_Window;
  APSDLRenderer: PSDL_Renderer; APSDLSurface: PSDL_Surface);
begin
  PSDLWindow := APSDLWindow;
  PSDLRenderer := APSDLRenderer;
  PSDLSurface  := APSDLSurface;
end;

function MonitorFromWindow: TSDL_Rect;
begin
  SDL_GetDisplayBounds(SDL_GetWindowDisplayIndex(PSDLWindow), @Result);
end;

procedure InvalidateWindow;
begin
  SDL_UpdateWindowSurface(PSDLWindow);
end;

procedure RaiseWindow;
begin
  SDL_RaiseWindow(PSDLWindow)
end;

function WindowDeviceContextHandle: THandle;
var
  LWindowInfo: TSDL_SysWMinfo;
begin
  if Assigned(PSDLWindow) then begin
    SDL_VERSION(LWindowInfo.version);
    if SDL_GetWindowWMInfo(PSDLWindow, @LWindowInfo) = SDL_TRUE then begin
      Result := LWindowInfo.win.hdc;
    end;
  end;
end;

function WindowHandle: THandle;
var
  LWindowInfo: TSDL_SysWMinfo;
begin
  if Assigned(PSDLWindow) then begin
    SDL_VERSION(LWindowInfo.version);
    if SDL_GetWindowWMInfo(PSDLWindow, @LWindowInfo) = SDL_TRUE then begin
      Result := LWindowInfo.win.window;
    end;
  end;
end;

function WindowBoundsRect: TSDL_Rect;
begin
  with Result do begin
    SDL_GetWindowPosition(PSDLWindow, @x, @y);
    SDL_GetWindowSize(PSDLWindow, @w, @h);
  end;
end;



end.

