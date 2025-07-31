{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.renderer.testmode;

{$mode ObjFPC}{$H+}

interface

procedure Initialize;
procedure Finalize;
procedure SaveFrame;

implementation

uses
  SysUtils,
  FileUtil,
  SDL2,
  sdl2_image,
  sdl.app.video.methods,
  session.pool,
  session.strutils,
  timestamps,
  timestamps.methods;

var
  GSurface : PSDL_Surface;
  GFormat: PSDL_PixelFormat;

procedure Initialize;
const
  LWitdh : LongInt = 32;
var
  LRect : TSDL_Rect;
begin
  GFormat := SDL_AllocFormat(SDL_PIXELFORMAT_ARGB8888);
  LRect := MonitorFromWindow;
  with GFormat^ do begin
    GSurface := SDL_CreateRGBSurface(
      0, LRect.w, LRect.h, LWitdh, Rmask, Gmask, Bmask, Amask);
  end;

  ForceDirectories(TestModeFolder);
end;

procedure Finalize;
begin
  SDL_FreeFormat(GFormat);
  SDL_FreeSurface(GSurface);
end;

procedure SaveFrame;
var
  LFilename : string;
  PFilename : PAnsiChar;
  function GetFileName : string;
  begin
    Result := ConcatPaths([TestModeFolder,
      'B' + Format('%.3d', [Pool.Session.Block.UID + 1])+ '-' +
      'T' + Format('%.3d', [Pool.Session.Trial.UID + 1])+ '-' +
      ClockMonotonic.ToString.Replace(',', '-') + '.png']);
  end;
begin
  SDL_RenderReadPixels(PSDLRenderer, nil, SDL_PIXELFORMAT_ARGB8888,
    GSurface^.pixels, GSurface^.pitch);
  LFilename := GetFileName;
  PFilename := PAnsiChar(LFilename);
  IMG_SavePNG(GSurface, PFilename);
end;

end.

