{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.controller.types;

{$mode ObjFPC}{$H+}

interface

uses Math;

type
  TControllerCode = (gcMouse, gcKeyboard, gcWii, gcPS4);

  TDirection = (None,
    Top,     Left,     Bottom,     Right,
    TopLeft, TopRight, BottomLeft, BottomRight);

  TPoint = record
    X : Float;
    Y : Float;
  end;

function CalculateAngleDegrees(APoint : TPoint): Float;
function GetDirection(Degree: Float): TDirection;

var
  SectorPad : Float;
  Sector0 : Float;
  Sector1 : Float;
  Sector2 : Float;
  Sector3 : Float;
  Sector4 : Float;
  Sector5 : Float;
  Sector6 : Float;
  Sector7 : Float;

implementation

const
  SectorSize: Float = 45;
  MaxDegress : Float = 360;
  HalfDegree : Float = 180;

function CalculateAngleDegrees(APoint: TPoint): Float;
var
  AngleRadians, AngleDegrees: Float;
begin
  AngleRadians := ArcTan2(APoint.Y, APoint.X);
  AngleDegrees := AngleRadians * (HalfDegree/PI);
  AngleDegrees := AngleDegrees + MaxDegress;
  Result := AngleDegrees - Trunc(AngleDegrees/MaxDegress) * MaxDegress;
end;

function GetDirection(Degree: Float): TDirection;
begin
  if      (Degree >= Sector0) or  (Degree < Sector1) then
    Result := TDirection.Right
  else if (Degree >= Sector1) and (Degree < Sector2) then
    Result := TDirection.BottomRight
  else if (Degree >= Sector2) and (Degree < Sector3) then
    Result := TDirection.Bottom
  else if (Degree >= Sector3) and (Degree < Sector4) then
    Result := TDirection.BottomLeft
  else if (Degree >= Sector4) and (Degree < Sector5) then
    Result := TDirection.Left
  else if (Degree >= Sector5) and (Degree < Sector6) then
    Result := TDirection.TopLeft
  else if (Degree >= Sector6) and (Degree < Sector7) then
    Result := TDirection.Top
  else if (Degree >= Sector7) and (Degree < Sector0) then
    Result := TDirection.TopRight;
end;

initialization
  SectorPad := Float(SectorSize)/Float(2);
  Sector0:= MaxDegress-SectorPad;
  Sector1:= (Sectorsize * 1)-SectorPad;
  Sector2:= (SectorSize * 2)-SectorPad;
  Sector3:= (SectorSize * 3)-SectorPad;
  Sector4:= (SectorSize * 4)-SectorPad;
  Sector5:= (SectorSize * 5)-SectorPad;
  Sector6:= (SectorSize * 6)-SectorPad;
  Sector7:= (SectorSize * 7)-SectorPad;


end.

