{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.graphics.debug;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Math;

  procedure PaintDebugGraphics;
  procedure DrawDebugCircle(ADegree : Float);

implementation

uses
  sdl2,
  sdl2_gfx,
  sdl.app.video.methods,
  sdl.app.graphics.text,
  sdl.app.controller.types,
  sdl.colors;

const
  CenterX = 150;
  CenterY = 150;
  Radius = 100;

var
  Text : TText = nil;
  Aim : TPoint;
  AimDegree : Float;

function CircumferencePoint(ADegree : Float) : TPoint;
var
  LAngleRadians: Float = 0;
begin
  LAngleRadians := ADegree * (Pi / Float(180));

  Result.X := CenterX + Radius * Cos(LAngleRadians);
  Result.Y := CenterY + Radius * Sin(LAngleRadians);
end;

procedure PaintDebugGraphics;
  procedure DrawLines(Color: TSDL_Color; ADegress : array of Float);
  var
    Degree : Float;
    Point : TPoint;
  begin
    for Degree in ADegress do begin
      Point := CircumferencePoint(Degree);
      with Color do begin
        aalineRGBA(PSDLRenderer,
              CenterX, CenterY, Round(Point.X), Round(Point.Y), r, g, b, a);
      end;
    end;
  end;

begin
  DrawLines(clLightBlueShaded1, [
    Sector0, Sector1, Sector2, Sector3,
    Sector4, Sector5, Sector6, Sector7]);

  case GetDirection(AimDegree) of
    Right:
      DrawLines(clBlack, [Sector0, Sector1]);
    BottomRight:
      DrawLines(clBlack, [Sector1, Sector2]);
    Bottom:
      DrawLines(clBlack, [Sector2, Sector3]);
    BottomLeft:
      DrawLines(clBlack, [Sector3, Sector4]);
    Left:
      DrawLines(clBlack, [Sector4, Sector5]);
    TopLeft:
      DrawLines(clBlack, [Sector5, Sector6]);
    Top:
      DrawLines(clBlack, [Sector6, Sector7]);
    TopRight:
      DrawLines(clBlack, [Sector7, Sector0]);
    None: { do nothing };
  end;


  with clRed do begin
    aacircleRGBA(PSDLRenderer,
      CenterX, CenterY, Radius, r, g, b, a);

    aalineRGBA(PSDLRenderer,
      CenterX, CenterY, Round(Aim.X), Round(Aim.Y), r, g, b, a);
  end;


  if Assigned(Text) then begin
    Text.AsIPaintable.Paint;
  end;
end;

procedure DrawDebugCircle(ADegree: Float);
begin
  if not Assigned(Text) then begin
    Text := TText.Create;
    Text.FontName := 'Arimo-Regular';
    Text.FontSize := 14;
    Text.Visible := True;
  end;
  Text.Load(ADegree.ToString);
  Text.Left := 100;
  Text.Top := 100;
  AimDegree := ADegree;
  Aim := CircumferencePoint(ADegree);
end;

finalization
  if Assigned(Text) then begin
    Text.Free;
  end;

end.

