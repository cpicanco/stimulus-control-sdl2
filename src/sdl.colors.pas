{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.colors;

{$mode ObjFPC}{$H+}

interface

uses SDL2, Graphics;

const
  clWhite : TSDL_Color =
    (r: 255; g: 255; b: 255; a: SDL_ALPHA_OPAQUE);

  clBlack : TSDL_Color =
    (r: 0;   g: 0;   b: 0;   a: SDL_ALPHA_OPAQUE);

  clTransparent : TSDL_Color =
    (r: 0;   g: 0;   b: 0;   a: SDL_ALPHA_TRANSPARENT);

  clRed   : TSDL_Color =
    (r: 255; g: 0;   b: 0;   a: SDL_ALPHA_OPAQUE);

  clGray  : TSDL_Color =
    (r: 128; g: 128; b: 128; a: SDL_ALPHA_OPAQUE);

  clDarkRed : TSDL_Color =
    (r: 60; g: 0;   b: 0;   a: SDL_ALPHA_OPAQUE);

  clLightBlueShaded1 : TSDL_Color =
    (r: 134; g: 150; b: 254; a: 50);

  clLightBlueShaded2 : TSDL_Color =
    (r: 134; g: 150; b: 254; a: 150);

  clLightRedShaded1 : TSDL_Color =
    (r: 255; g: 57; b: 71; a: 25);

  clLightGrayShaded1 : TSDL_Color =
    (r: 128; g: 128; b: 128; a: 100);

function ColorToSDLColor(AColor: TColor): TSDL_Color;

var
  clBackground : TSDL_Color;
  clTimeOut : TSDL_Color;
  clFontColor : TSDL_Color;
  clDefaultBackground : TSDL_Color;

implementation

function ColorToSDLColor(AColor: TColor): TSDL_Color;
var
  RGBColor: LongInt;
begin
  RGBColor := ColorToRGB(AColor);

  Result.r := (RGBColor and $000000FF);          // Red component
  Result.g := (RGBColor and $0000FF00) shr 8;    // Green component
  Result.b := (RGBColor and $00FF0000) shr 16;   // Blue component
  Result.a := SDL_ALPHA_OPAQUE;                  // Set alpha to opaque
end;

initialization
  clDefaultBackground := clWhite;
  clBackground := clWhite;
  clTimeOut := clDarkRed;
  clFontColor := clBlack;

end.

