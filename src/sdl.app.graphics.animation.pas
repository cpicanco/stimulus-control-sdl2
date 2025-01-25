{
  Stimulus Control
  Copyright (C) 2025-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.graphics.animation;

{$mode objfpc}{$H+}

interface

uses
  SysUtils
  , SDL2
  , sdl2_gfx
  , sdl.app.graphics.rectangule
  , sdl.app.paintable.contract
  , sdl.app.grids.types
  , sdl.colors
  ;

type
  { TAnimation }

  TAnimation = class(TRectangule, IPaintable)
  private
    FSibling : TRectangule;
    FVisible: Boolean;
  protected
    procedure Paint; override;
  public
    constructor Create; override;
    destructor Destroy; override;
    procedure Animate(ASibling : TRectangule);
    procedure Join(ASample, AComparison : TRectangule;
      AGridOrientation : TGridOrientation);
    procedure Stop;
    procedure Hide; override;
    property Sibling : TRectangule read FSibling;
    property Visible : Boolean read FVisible write FVisible;
  end;

implementation

uses
  LazFileUtils, Math
  , sdl.app.video.methods
  ;

type

  TAnimationData = record
    Acum: Float;
    Growing: boolean;
    Step: Float;
    FixedHeight : integer;
    MinHeight : integer;
    MaxHeight : integer;
    MinWidth  : integer;
    MaxWidth  : integer;
  end;

var
  AnimationData : TAnimationData;

{ TAnimation }

procedure TAnimation.Paint;
var
  TempSize: Float;
  function easeInOutQuad(t: Float): Float;
  begin
    if t < 0.5 then
      Result := 2 * t * t
    else
      Result := -1 + (4 - 2 * t) * t;
  end;

  procedure Line(x1, y1, x2, y2 : LongInt);
  begin
    with clRed do begin
      thickLineRGBA(PSDLRenderer,
        x1, y1, x2, y2,
        4, r, g, b, a)
    end;
  end;

  procedure Square(ARect : TSDL_Rect);
  begin
    with ARect do begin
      Line(x, y, x+w, y);
      Line(x+w, y, x+w, y+h);
      Line(x+w, y+h, x, y+h);
      Line(x, y+h, x, y);
    end;
  end;

begin
  if Assigned(FSibling) and FVisible then begin
    with AnimationData do begin
      Acum := Acum + Step;
      if Step > 1 then
        Step := 1;
      TempSize := easeInOutQuad(Acum);
      if Growing then
      begin
        Height := Round(FixedHeight * TempSize);
        Width := Height;
        if Height >= FixedHeight then
        begin
          Height := FixedHeight;
          Width := Height;
          Growing := False;
          Acum:= 0;
        end;
      end else begin
        TempSize := FixedHeight - Round(FixedHeight * TempSize);
        if TempSize <= MinHeight then
        begin
          Height := MinHeight;
          Width := MinWidth;
          Growing := true;
          Acum:= 0;
        end else begin
          Height := Trunc(TempSize);
          Width := Height;
        end;
      end;
    end;
    CentralizeWith(FSibling.BoundsRect);
    Square(FRect);
  end;
end;

procedure TAnimation.Animate(ASibling : TRectangule);
begin
  FSibling := ASibling;
  FRect := ASibling.BoundsRect;
  Inflate(10);

  AnimationData.MinHeight := ASibling.BoundsRect.h;
  AnimationData.MinWidth := ASibling.BoundsRect.w;
  AnimationData.FixedHeight := FRect.h + ((FRect.h*10) div 100);
  FVisible := True;
end;

procedure TAnimation.Join(ASample, AComparison: TRectangule;
  AGridOrientation : TGridOrientation);
begin
  Stop;

  //ASample.EdgeColor:=clInactiveCaption;
  case AGridOrientation of
    goNone : begin
        { do something }
    end;
    goTopToBottom : begin
      Top := ASample.Top -10;
      Left := ASample.Left -15;
      Height := ASample.Height + AComparison.Height + 30;
      Width := ASample.Width + 30;
    end;
    goBottomToTop : begin
      Top := AComparison.Top -10;
      Left := AComparison.Left -15;
      Height := AComparison.Height + ASample.Height + 30;
      Width := AComparison.Width + 30;
    end;
    goLeftToRight : begin
      Top := ASample.Top -15;
      Left := ASample.Left -10;
      Width := ASample.Width + AComparison.Width + 30;
      Height := ASample.Height + 30;
    end;
    goRightToLeft : begin
      Top := AComparison.Top -15;
      Left := AComparison.Left -10;
      Width := AComparison.Width + ASample.Width + 30;
      Height := AComparison.Height + 30;
    end;
    otherwise begin

    end;
  end;
end;

procedure TAnimation.Stop;
begin
  // change color
end;

procedure TAnimation.Hide;
begin
  inherited Hide;
  FVisible := False;
end;

constructor TAnimation.Create;
begin
  inherited Create;
  AnimationData.Step := 0.025; // for 50 fps
  //FPenWidth := 6;
  FVisible := False;
  FSibling := nil;
end;

destructor TAnimation.Destroy;
begin
  inherited Destroy;
end;

end.

