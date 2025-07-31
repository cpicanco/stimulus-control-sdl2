{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.controller.axis;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, SDL2, sdl.app.controller.types;

type

  { TController2DAxis }

  TController2DAxis = class
    private
      FGameController : PSDL_GameController;
      FCurrentDirection : TDirection;
      FLastDirection: TDirection;
      FPoint : TPoint;
    public
      constructor Create;
      destructor Destroy; override;
      function IsOutSideDeadZone : Boolean;
      procedure UpdateCurrentDirection;
      property GameController : PSDL_GameController
        read FGameController write FGameController;
      property CurrentDirection : TDirection read FCurrentDirection;
      property LastDirection : TDirection read FLastDirection write FLastDirection;
  end;

implementation

uses
  sdl.app.graphics.debug,
  Math;

const
  DeadZoneThreshold = 0.2;

// https://github.com/Minimuino/thumbstick-deadzones

function MapRange(AValue, OldMin, OldMax, NewMin, NewMax: Float): Float;
begin
  Result :=
    (NewMin + (NewMax - NewMin)) * ((AValue - OldMin) / (OldMax - OldMin));
end;

function SlopedScaledAxialDeadzone(AX, AY: Float): TPoint;
var
  LX, LY, DeadzoneX, DeadzoneY: Float;
  SignX: Float;
  SignY: Float;
begin
  LX := 0;
  LY := 0;
  DeadzoneX := DeadZoneThreshold * Power(Abs(AX), 2);
  DeadzoneY := DeadZoneThreshold * Power(Abs(AY), 2);
  SignX := Sign(AX);
  SignY := Sign(AY);

  if Abs(AX) > DeadzoneX then
    LX := SignX * MapRange(Abs(AX), DeadzoneX, 1, 0, 1);

  if Abs(AY) > DeadzoneY then
    LY := SignY * MapRange(Abs(AY), DeadzoneY, 1, 0, 1);

  Result.X := LX;
  Result.Y := LY;
end;

function ScaledRadialDeadzone(AX, AY : Float): TPoint;
var
  Magnitude: Float;
  NormalizedX : Float;
  NormalizedY : Float;
begin
  Magnitude := Hypot(AX, AY);

  if Magnitude < DeadZoneThreshold then begin
    Result.X := 0;
    Result.Y := 0;
  end else begin
    NormalizedX := AX / Magnitude;
    NormalizedY := AY / Magnitude;

    Result.X := NormalizedX * MapRange(Magnitude, DeadZoneThreshold, 1, 0, 1);
    Result.Y := NormalizedY * MapRange(Magnitude, DeadZoneThreshold, 1, 0, 1);
  end;
end;

function IsOutDeadZone(var AX : Float; var AY : Float) : Boolean;
var
  Magnitude: Float;
  Output : TPoint;
begin
  Magnitude := Hypot(AX, AY);

  if Magnitude < DeadZoneThreshold then begin
    AX := 0;
    AY := 0;
    Result := False;
  end else begin
    Output := ScaledRadialDeadzone(AX, AY);
    Output := SlopedScaledAxialDeadzone(Output.X, Output.Y);
    AX := Output.X;
    AY := Output.Y;
    Result := True;
  end;
end;

{ TController2DAxis }

constructor TController2DAxis.Create;
begin
  FGameController := nil;
  FCurrentDirection := None;
  FLastDirection := None;
  FPoint.X := 0;
  FPoint.Y := 0;
end;

destructor TController2DAxis.Destroy;
begin
  inherited Destroy;
end;

function TController2DAxis.IsOutSideDeadZone: Boolean;
begin
  FPoint.X := Float(SDL_GameControllerGetAxis(
    FGameController, SDL_CONTROLLER_AXIS_LEFTX))/Float(MaxSmallint);

  FPoint.Y := Float(SDL_GameControllerGetAxis(
    FGameController, SDL_CONTROLLER_AXIS_LEFTY))/Float(MaxSmallint);

  Result := IsOutDeadZone(FPoint.X, FPoint.Y);
  //if Result then begin
  //  DrawDebugCircle(CalculateAngleDegrees(FPoint));
  //end;
end;

procedure TController2DAxis.UpdateCurrentDirection;
begin
  FCurrentDirection := GetDirection(CalculateAngleDegrees(FPoint));
end;


end.

