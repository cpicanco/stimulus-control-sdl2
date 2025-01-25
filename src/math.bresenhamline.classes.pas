{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço, Universidade Federal do Pará.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit math.bresenhamline.classes;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Math, Math.BresenhamLine, SDL2;

type

  { TBresenhamLine }

  TBresenhamLine = class(TObject)
  private
    FBresenhamLine : TPoints;
    FIndex         : integer;
  public
    constructor Create; overload;
    constructor Create(AOrigin, ADestin : TRect); overload;
    destructor Destroy; override;
    procedure Update(AOrigin, ADestin : TRect); overload;
    procedure Update(AOrigin, ADestin : TSDL_Rect); overload;
    function NextPoint : TPoint;
    function GetPoint(AValue : Float) : TPoint;
    property Points : TPoints read FBresenhamLine;
  end;


var
  ChannelDragMouseMoveFactor : integer;

implementation

{ TBresenhamLine }

constructor TBresenhamLine.Create;
begin
  FIndex := 0;
end;

constructor TBresenhamLine.Create(AOrigin, ADestin: TRect);
begin
  Update(AOrigin, ADestin);
end;

destructor TBresenhamLine.Destroy;
begin
  SetLength(FBresenhamLine, 0);
  inherited Destroy;
end;

//procedure TBresenhamLine.Paint;
//var
//  Point : TPoint;
//begin
//  if Length(Line) > 0 then begin
//    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
//    for Point in FBresenhamLine do
//      SDL_RenderDrawPoint(Point.X, Point.Y);
//  end;
//end;

procedure TBresenhamLine.Update(AOrigin, ADestin: TRect);
begin
  FIndex := 0;
  FBresenhamLine := BresenhamLine(
    AOrigin.Left,
    ADestin.Left,
    AOrigin.Top,
    ADestin.Top);
end;

procedure TBresenhamLine.Update(AOrigin, ADestin: TSDL_Rect);
var
  LOrigin, LDestin: TRect;
begin
  with AOrigin do begin
    LOrigin := Rect(x, y, x+w, y+h);
  end;

  with ADestin do begin
    LDestin := Rect(x, y, x+w, y+h);
  end;

  Update(LOrigin, LDestin);
end;

function TBresenhamLine.NextPoint: TPoint;
var
  LLength : integer;
begin
  LLength := Length(FBresenhamLine);
  if LLength > 0 then
    Result := FBresenhamLine[FIndex];
  if FIndex < LLength-1 then
    Inc(FIndex, ChannelDragMouseMoveFactor);

  if FIndex >= LLength then
    FIndex := LLength-1;
end;

function TBresenhamLine.GetPoint(AValue: Float): TPoint;
var
  LLength: Float;
  LLineIndex : Float;
begin
  LLength := High(FBresenhamLine);
  LLineIndex := AValue * LLength / 100;
  Result := FBresenhamLine[Round(LLineIndex)];
end;

end.

