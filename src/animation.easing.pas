{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit animation.easing;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Math;

type

  TData = record
    AccumulatedSteps: Float;
    IsGrowing: Boolean;
    Step: Float;
    CurrentValue : Byte;
    FixedValue : Integer;
    MinValue : Integer;
    MaxValue : Integer;
  end;

  { TEasingAnimation }

  TEasingAnimation = class
  private
    FData: TData;
    function GetValue: Byte;
  public
    constructor Create;
    destructor Destroy; override;
    procedure StepIt;
    procedure Reset;
    property Value : Byte read GetValue;
  end;

implementation

{ TEasingAnimation }

function TEasingAnimation.GetValue: Byte;
begin
  Result:= FData.CurrentValue;
end;

constructor TEasingAnimation.Create;
begin
  inherited Create;
  with FData do begin
    AccumulatedSteps := 0;
    CurrentValue := 0;
    IsGrowing := False;

    Step := 0.025;
    MinValue := 0;
    MaxValue := 255;
    FixedValue := 125;
  end;
end;

destructor TEasingAnimation.Destroy;
begin
  inherited Destroy;
end;

procedure TEasingAnimation.StepIt;
var
  TempSize: Float;
  function easeInOutQuad(t: Float): Float;
  begin
    if t < 0.5 then
      Result := 2 * t * t
    else
      Result := -1 + (4 - 2 * t) * t;
  end;
begin
  with FData do begin
    if Step > 1 then
       Step := 1;
    AccumulatedSteps := AccumulatedSteps + Step;
    TempSize := easeInOutQuad(AccumulatedSteps);
    if IsGrowing then begin
      CurrentValue := Round(FixedValue * TempSize);
      if CurrentValue >= FixedValue then
      begin
        CurrentValue := FixedValue;
        IsGrowing := False;
        AccumulatedSteps:= 0;
      end;
    end else begin
      TempSize := FixedValue - Round(FixedValue * TempSize);
      if TempSize <= MinValue then
      begin
        CurrentValue := MinValue;
        IsGrowing := True;
        AccumulatedSteps:= 0;
      end else begin
        CurrentValue := Trunc(TempSize);
      end;
    end;
  end;
end;

procedure TEasingAnimation.Reset;
begin
  with FData do begin
    AccumulatedSteps := 0;
    CurrentValue := 0;
    IsGrowing := False;
  end;
end;

end.

