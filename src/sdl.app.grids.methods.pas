{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.grids.methods;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math,
  sdl.app.grids.types;

function GetCentralRect(AWidth,AHeight,ASize:integer):TRect; overload;
function GetCentralRect(AWidth,AHeight,ALeftBorder,ATopBorder,
  ARightBorder,ABottomBorder: integer): TRect; overload;

// 0 degrees = rect right; increments clockwise, squares only
function GetPointFromAngle(AAngle: float; ARect:TRect): TPoint;
function GetCentralGrid(AN: integer; ASquareSide: Float;
  ADistribute: Boolean): TMatrix;
procedure InitMonitor;
function GetCircularCentralGrid(AN: integer; ASquareSide: Float): TMatrix;
function Border : TBorder;

implementation

uses
  SDL2
  , session.parameters.global
  {$IFNDEF GRIDS_TEST}, sdl.app{$ENDIF}
  ;

var
  ScreenInCentimeters : Float;
  MonitorWidth : integer;
  MonitorHeight: integer;
  BorderTop    : TSDL_Rect;
  BorderBottom : TSDL_Rect;
  BorderLeft   : TSDL_Rect;
  BorderRight  : TSDL_Rect;

procedure InitMonitor;
var
  LRect : TSDL_Rect;
begin
  ScreenInCentimeters := GlobalTrialParameters.ScreenInCentimeters;
  {$IFNDEF GRIDS_TEST}
  if Assigned(SDLApp) then begin
    LRect := SDLApp.Monitor;
  end else begin
    SDL_GetDisplayBounds(0, @LRect)
  end;
  {$ELSE}
   SDL_GetDisplayBounds(0, @LRect);
  {$ENDIF}
  MonitorWidth := LRect.w;
  MonitorHeight:= LRect.h;
end;

function GetCentralRect(AWidth, AHeight, ASize:integer): TRect;
begin
  Result := GetCentralRect(AWidth,AHeight,ASize,ASize,ASize,ASize);
end;

function GetCentralRect(AWidth, AHeight, ALeftBorder, ATopBorder, ARightBorder,
  ABottomBorder: integer): TRect;
var
  LSide : integer;
begin
  if AHeight > AWidth then
    begin
      LSide := AWidth;
      Result := Rect( (0 + ALeftBorder),
                      (0  + (AHeight div 2) - (LSide div 2) +  ATopBorder),
                      (LSide - ARightBorder),
                      (LSide + (AHeight div 2) - (LSide div 2) +  - ABottomBorder)
                    );
    end
  else if AHeight < AWidth then
    begin
      LSide := AHeight;
      Result := Rect( (0 + (AWidth div 2) - (LSide div 2) + ALeftBorder),
                      (0 + ATopBorder),
                      (LSide + (AWidth div 2) - (LSide div 2) - ARightBorder),
                      (LSide - ABottomBorder)
                    );
    end
  else if AHeight = AWidth then
    begin
      LSide := AHeight;
      Result := Rect( (0 + ALeftBorder),
                      (0 + ATopBorder),
                      (LSide - ARightBorder),
                      (LSide - ABottomBorder)
                    );
    end;
end;

function GetPointFromAngle(AAngle: float; ARect:TRect): TPoint;
var
  LRadius : float;
  LCenterX : float;
  LCenterY : float;
begin
  // http://math.stackexchange.com/questions/143932/calculate-point-given-x-y-angle-and-distance
  LRadius := (ARect.Right - ARect.Left)/ 2;
  LCenterX := LRadius + ARect.Left;
  LCenterY := LRadius + ARect.Top;
  Result.X :=  Round((LRadius * cos(DegtoRad(AAngle))) + LCenterX);
  Result.Y :=  Round((LRadius * sin(DegtoRad(AAngle))) + LCenterY);
end;

{
  GetPositionFromSegment returns Left or Top position based on:
    ASegment = Width or Height from which get the Left or Top position.
    ASteps = Desired number os columns or rows.
    AStep = Target column or row.
    AStimulusSide = Width or height of the target stimulus.
    AInterStimulusSpace = Desired horizontal or vertical space from one stimulus to another.
}
function GetPositionFromSegment(ASegment, AStep, ASteps,
  AStimulusSide, AInterStimulusSpace : integer):integer;
var
  LSize : integer;
begin
  LSize := AStimulusSide + AInterStimulusSpace;
  Result := Round((LSize*AStep)-((LSize*ASteps)/2)+((ASegment+AInterStimulusSpace)/2));
end;

function CmToScreenPixels(AMeasure : Float) : integer;
begin
  Result := Round(AMeasure*(MonitorWidth/ScreenInCentimeters));
end;

procedure SetBorders(ASize: integer);
begin
  with BorderTop do begin
    x := 0;
    y := 0;
    w := MonitorWidth;
    h := ASize;
  end;

  with BorderBottom do begin
    x := 0;
    y := BorderTop.h + MonitorHeight-(ASize*2);
    w := MonitorWidth;
    h := MonitorHeight;
  end;

  with BorderLeft do begin
    x := 0;
    y := 0;
    w := ASize;
    h := MonitorHeight;
  end;

  with BorderRight do begin
    x := BorderLeft.w + MonitorWidth-(ASize*2);
    y := 0;
    w := MonitorWidth;
    h := MonitorHeight;
  end;
end;

{Cria grade quadrada como uma matriz AN x AN. Quando ADistribute = true, a
distância horizontal e vertical entre os estímulos é diferente, e quando false
é igual}
function GetCentralGrid(AN: integer; ASquareSide: Float;
  ADistribute: Boolean): TMatrix;
var
  LIndex      : integer = 0;
  //LSegment    : integer = 0;
  //LSteps      : integer = 0;
  //LStep       : integer = 0;
  LSquareSide : integer = 0;
  LInterSpaceW : integer = 0;
  LInterSpaceH : integer = 0;
  j : integer = 0;
  i : integer = 0;
begin
  Result := Default(TMatrix);
  SetLength(Result, AN, AN);
  LSquareSide := CmToScreenPixels(ASquareSide);
  if ADistribute then begin
    LInterSpaceW := (MonitorWidth -  (LSquareSide * AN)) div AN;
    LInterSpaceH := (MonitorHeight - (LSquareSide * AN)) div AN;
  end else begin
    if MonitorWidth > MonitorHeight then begin
      LInterSpaceH := (MonitorHeight - (LSquareSide * AN)) div AN;
      LInterSpaceW := LInterSpaceH;
    end else begin
      LInterSpaceW := (MonitorWidth -  (LSquareSide * AN)) div AN;
      LInterSpaceH := LInterSpaceW;
    end;
  end;
  for j := Low(Result) to High(Result) do begin
    for i := Low(Result[j]) to High(Result[j]) do begin
      with Result[j][i] do begin
        Index := LIndex;
        Rect.y := Min(Max(GetPositionFromSegment(
          MonitorHeight, j, AN, LSquareSide, LInterSpaceH), 0),
          MonitorHeight - LSquareSide);
        Rect.x := Min(Max(GetPositionFromSegment(
          MonitorWidth, i, AN, LSquareSide, LInterSpaceW), 0),
          MonitorWidth - LSquareSide);
        Rect.w := LSquareSide;
        Rect.h := LSquareSide;
      end;
      Inc(LIndex);
    end;
  end;
  SetBorders(Result[0][0].Rect.y);
end;

{Cria grade circular considerando j como modelo central e i como comparações em
torno de um diâmetro. AN = número de estímulos i; ASquareSide = lado do quadrado
dos estímulos}
function GetCircularCentralGrid(AN: integer; ASquareSide: Float): TMatrix;
var
  LIndex      : integer = 0;
  //LSegment    : integer = 0;
  //LSteps      : integer = 0;
  //LStep       : integer = 0;
  LSquareSide : integer = 0;
  LDegree : integer = 0;
  LDegreeI : integer = 0;
  LPoint : TPoint;
  LRect  : TRect;
  j : integer = 0;
  i : integer = 0;
const
  BaseDegree : integer = 360;
begin
  Result := Default(TMatrix);
  SetLength(Result, 2);
  SetLength(Result[0], AN);
  SetLength(Result[1], 1);
  LSquareSide := CmToScreenPixels(ASquareSide);
  SetBorders(LSquareSide div 2);
  LDegree := BaseDegree;
  LDegreeI := BaseDegree div AN;
  LRect := GetCentralRect(MonitorWidth, MonitorHeight, LSquareSide div 2);
  for j := Low(Result) to High(Result) do begin
    for i := Low(Result[j]) to High(Result[j]) do begin
      with Result[j][i] do begin
        case j of
          0:  begin
            Index := LIndex;
            LPoint := GetPointFromAngle(LDegree, LRect);
            Rect.y := LPoint.Y - (LSquareSide div 2);
            Rect.x := LPoint.X - (LSquareSide div 2);
            Rect.w := LSquareSide;
            Rect.h := LSquareSide;
            Inc(LDegree, LDegreeI);
          end;

          1: begin
            Index := LIndex;
            Rect.y  := (MonitorHeight div 2) - (LSquareSide div 2);
            Rect.x := (MonitorWidth div 2) - (LSquareSide div 2);
            Rect.w := LSquareSide;
            Rect.h := LSquareSide;
          end;
        end;
      end;
      Inc(LIndex);
    end;
  end;
end;

function Border: TBorder;
begin
  with Result do begin
    Top := BorderTop;
    Bottom := BorderBottom;
    Left := BorderLeft;
    Right := BorderRight;
  end;
end;


//{
//  3x3
//  0..1..2
//  3..4..5
//  6..7..8
//}
//function IntToCell(AN: Integer) : TCell;
//begin
//  case AN of
//    0 : Result := [0, 0];
//    1 : Result := [0, 1];
//    2 : Result := [0, 2];
//    3 : Result := [1, 0];
//    4 : Result := [1, 1];
//    5 : Result := [1, 2];
//    6 : Result := [2, 0];
//    7 : Result := [2, 1];
//    8 : Result := [2, 2];
//  end;
//end;

end.


