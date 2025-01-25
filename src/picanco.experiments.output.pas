{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit picanco.experiments.output;

{$mode ObjFPC}{$H+}

interface

uses
  Classes,
  SysUtils,
  Graphics,
  picanco.experiments.constants,
  picanco.experiments.words.types;

procedure InitializeBitmap;
procedure FinalizeBitmap;
procedure PrintWordsPerCycle;
procedure ShowWordsPerCycle;

var
  Bitmap : TBitmap;

implementation

uses
  forms.canvas.playground
  //, sdl.app.output
  , session.strutils;

const
  LSpace : integer = 10;
  LSize  : integer = 100;

procedure PrintWordsPerCycle;
var
  E1Cycle : TCycle;
  E1Code  : TAlphaNumericCode;
  LCycle, LCode, LWord, LLine : string;
begin
  for E1Cycle := Low(E1CyclesRange) to High(E1CyclesRange) do begin
    WriteStr(LCycle, E1Cycle);
    //Print(LCycle);
    LLine := '';
    for E1Code := Low(E1CyclesCodeRange) to High(E1CyclesCodeRange) do begin
      WriteStr(LCode, E1Code);
      LWord := E1WordPerCycleCode[E1Cycle, E1Code];
      LLine := LLine + String.Join('=', [LCode, LWord]) + ',';
    end;
    //Print(Copy(LLine, 0, (length(LLine)-1)));
  end;
end;

procedure ShowWordsPerCycle;
begin
  FormCanvas := TFormCanvas.Create(nil);
  try
    FormCanvas.ShowModal;
  finally
    FormCanvas.Free;
  end;
end;

function DrawText(Canvas: TCanvas; APoint: TPoint; ASize: integer;
  AFontName, AWord : string) : TPoint;
var
  LOldFontName : string;
begin
  LOldFontName := Canvas.Font.Name;

  Canvas.Font.Name := AFontName;
  Canvas.Font.Size := ASize;
  Canvas.TextOut(APoint.X, APoint.Y, AWord);

  Result.X := APoint.X+Canvas.TextExtent(AWord).Width;
  Result.Y := APoint.Y+Canvas.TextExtent(AWord).Height;
  Canvas.Font.Name := LOldFontName;
end;


function DrawEquivalenceClassLine(Canvas: TCanvas; APoint: TPoint;
  AWord: string; HasImage : Boolean) : TPoint;
var
  LFile : string;
  LImage : TPortableNetworkGraphic;
  LPoint1 : TPoint;
  LPoint2 : TPoint;
  LRect : TRect;
begin
  LImage := TPortableNetworkGraphic.Create;
  try
    LFile := AsImage(AWord)+'.png';
    if HasImage then begin
      LImage.LoadFromFile(LFile);
    end else begin
      LImage.SetSize(LSize, LSize);
      LImage.Canvas.Pen.Width := 10;
      LImage.Canvas.Pen.Color := clRed;
      LImage.Canvas.Brush.Color := clWhite;
      LImage.Canvas.FillRect(0, 0, LSize, LSize);
      LImage.Canvas.Line(0, 0, LSize, LSize);
      LImage.Canvas.Line(0, LSize, LSize, 0);
    end;
    LRect := Rect(APoint.X, APoint.Y, APoint.X+LSize, APoint.Y+LSize);
    Canvas.StretchDraw(LRect, LImage);
    Result.X := APoint.X+LSize+LSpace;
    Result.Y := APoint.Y;

    LPoint1 := DrawText(Canvas, Result, 25, 'Arimo Regular', AWord);
    //Result.X := LPoint1.X + LSpace;
    Result.Y := LPoint1.Y + LSpace;

    LPoint2 := DrawText(Canvas, Result, 40, 'Picanco et al', AWord);
    Result.X := LPoint2.X;

    Result.Y := APoint.Y+LSize;
  finally
    LImage.Free;
  end;
end;

procedure DrawEquivalenceClasses(Canvas: TCanvas);
var
  E1Cycle : TCycle;
  E1Code  : TAlphaNumericCode;
  LCycle, LCode, LWord : string;
  CurrentPoint : TPoint;
  LB : Boolean;
  LX : integer;
begin
  CurrentPoint.X := 0;
  LX := CurrentPoint.X;
  for E1Cycle := Low(E1CyclesRange) to High(E1CyclesRange) do begin

    WriteStr(LCycle, E1Cycle);
    CurrentPoint.X := LX;
    CurrentPoint.Y := 0;

    CurrentPoint := DrawText(Canvas, CurrentPoint, 12, 'Arimo-Regular', LCycle);
    CurrentPoint.Y := CurrentPoint.Y+LSpace;

    for E1Code := Low(E1CyclesCodeRange) to High(E1CyclesCodeRange) do begin
      WriteStr(LCode, E1Code);
      LWord := E1WordPerCycleCode[E1Cycle, E1Code];
      CurrentPoint.X := LX;

      LB := WordHasImage(LWord);
      CurrentPoint := DrawEquivalenceClassLine(Canvas, CurrentPoint, LWord, LB);
      CurrentPoint.Y := CurrentPoint.Y+LSpace;
    end;
    LX := CurrentPoint.X + LSpace;
  end;

end;

procedure InitializeBitmap;
begin
  Bitmap := TBitmap.Create;
  with Bitmap do begin
    SetSize((LSize+LSpace)*2*7, ((LSize+LSpace)*6)+(12+LSpace));
    Canvas.Brush.Color := clWhite;
    Canvas.Pen.Color := clBlack;
    Canvas.FillRect(0, 0, Width, Height);
    DrawEquivalenceClasses(Canvas);
  end;
end;

procedure FinalizeBitmap;
begin
  Bitmap.Free;
end;

end.

