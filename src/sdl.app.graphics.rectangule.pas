{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.graphics.rectangule;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , SDL2
  , ctypes
  , sdl.app.events.abstract
  , sdl.app.controls.custom
  ;

type

  { TRectangule }

  TRectangule = class(TSDLControl)
  private
    FCanShade : Boolean;
    FShaded : Boolean;
    FOriginalBounds : TSDL_Rect;
    FEdgeColor: TSDL_Color;
    FVisible: Boolean;
    function GetHeight: cint;
    function GetLeft: cint;
    function GetTop: cint;
    function GetWidth: cint;
    procedure SetHeight(AValue: cint);
    procedure SetLeft(AValue: cint);
    procedure SetTop(AValue: cint);
    procedure SetWidth(AValue: cint);
  protected
    procedure MouseUp(Sender: TObject; Shift: TCustomShiftState;
      X, Y: Integer); override;
    procedure MouseEnter(Sender: TObject); override;
    procedure MouseExit(Sender: TObject); override;
    procedure GazeEnter(Sender: TObject); override;
    procedure GazeExit(Sender: TObject); override;
    procedure Paint; override;
  public
    constructor Create; override;
    destructor Destroy; override;
    function IntersectsWith(ARect: TSDL_Rect) : Boolean; overload;
    function IntersectsWith(ARect: TRectangule) : Boolean; overload;
    function GetOriginalBounds : TSDL_Rect;
    procedure Centralize;
    procedure CentralizeAtTopWith(ARect: TSDL_Rect);
    procedure MoveToBottomRightScreen;
    procedure CentralizeAtRightWith(ARect: TSDL_Rect; AFactor : Byte);
    procedure CentralizeWith(ARect: TSDL_Rect);
    procedure Fit(ARect : TSDL_Rect);
    procedure DoRandomMouseUp;
    procedure Inflate(AValue : cint);
    procedure Hide; virtual;
    procedure ToOriginalBounds;
    procedure SetOriginalBounds;
    procedure Show;
    property BoundsRect : TSDL_Rect read GetBoundsRect write SetBoundsRect;
    property CanShade : Boolean read FCanShade write FCanShade;
    property EdgeColor: TSDL_Color read FEdgeColor write FEdgeColor;
    property Left   : cint read GetLeft write SetLeft;
    property Height : cint read GetHeight write SetHeight;
    property Top    : cint read GetTop write SetTop;
    property Width  : cint read GetWidth write SetWidth;
    property Visible : Boolean read FVisible write FVisible;
  end;

implementation

uses sdl.app.video.methods, sdl.colors, math, sdl.app.testmode;

{ TStimulus }

function TRectangule.GetHeight: cint;
begin
  Result := FRect.h;
end;

function TRectangule.GetLeft: cint;
begin
  Result := FRect.x;
end;

function TRectangule.GetTop: cint;
begin
  Result := FRect.y;
end;

function TRectangule.GetWidth: cint;
begin
  Result := FRect.w;
end;

procedure TRectangule.MouseUp(Sender: TObject; Shift: TCustomShiftState; X,
  Y: Integer);
begin
  if Visible then begin
    inherited MouseUp(Sender, Shift, X, Y);
  end;
end;

procedure TRectangule.SetHeight(AValue: cint);
begin
  if FRect.h=AValue then Exit;
  FRect.h := AValue;
end;

procedure TRectangule.SetLeft(AValue: cint);
begin
  if FRect.x=AValue then Exit;
  FRect.x:=AValue;
end;

procedure TRectangule.SetTop(AValue: cint);
begin
  if FRect.y=AValue then Exit;
  FRect.y:=AValue;
end;

procedure TRectangule.SetWidth(AValue: cint);
begin
  if FRect.w=AValue then Exit;
  FRect.w:=AValue;
end;

procedure TRectangule.MouseEnter(Sender: TObject);
begin
  if Visible then begin
    inherited MouseEnter(Sender);
    FShaded := True;
  end;
end;

procedure TRectangule.MouseExit(Sender: TObject);
begin
  if Visible then begin
    inherited MouseExit(Sender);
    FShaded := False;
  end;
end;

procedure TRectangule.GazeEnter(Sender: TObject);
begin
  if Visible then begin
    inherited GazeEnter(Sender);
    FShaded := True;
  end;
end;

procedure TRectangule.GazeExit(Sender: TObject);
begin
  if Visible then begin
    inherited GazeExit(Sender);
    FShaded := False;
  end;
end;

procedure TRectangule.Paint;
begin
  if Visible then begin
    if FCanShade and FShaded then begin
      SDL_SetRenderDrawBlendMode(PSDLRenderer, SDL_BLENDMODE_BLEND);
      with clLightBlueShaded1 do
        SDL_SetRenderDrawColor(PSDLRenderer, r, g, b, a);
      SDL_RenderFillRect(PSDLRenderer, @FRect);
    end;
  end else begin
    if TestMode then begin
      with clGray do
        SDL_SetRenderDrawColor(PSDLRenderer, r, g, b, a);
      SDL_RenderFillRect(PSDLRenderer, @FRect);
    end;
  end;
end;

constructor TRectangule.Create;
begin
  inherited Create;
  FCanShade:= True;
  FShaded  := False;
  FVisible := False;
  with FRect do begin
    x := 0;
    y := 0;
    w := 200;
    h := 200;
  end;
  SetOriginalBounds;
  EdgeColor := clRed
end;

destructor TRectangule.Destroy;
begin
  { free stuff }
  inherited Destroy;
end;

procedure TRectangule.Centralize;
begin
  Left := (Parent.BoundsRect.w div 2) - (Width  div 2);
  Top  := (Parent.BoundsRect.h div 2) - (Height div 2);
end;

procedure TRectangule.Show;
begin
  FVisible := True;
end;

procedure TRectangule.Hide;
begin
  FVisible := False;
end;

function TRectangule.IntersectsWith(ARect: TSDL_Rect): Boolean;
begin
  Result := SDL_HasIntersection(@FRect, @ARect);
end;

function TRectangule.IntersectsWith(ARect: TRectangule): Boolean;
begin
  Result := IntersectsWith(ARect.BoundsRect);
end;

function TRectangule.GetOriginalBounds: TSDL_Rect;
begin
  Result := FOriginalBounds;
end;

procedure TRectangule.CentralizeWith(ARect: TSDL_Rect);
begin
  Left := ARect.x + (ARect.w div 2) - (Width  div 2);
  Top  := ARect.y + (ARect.h div 2) - (Height div 2);
end;

procedure TRectangule.Fit(ARect: TSDL_Rect);
const
  Border = 20;
var
  ParentX, ParentY: Integer;
begin
  ParentX := 0;
  ParentY := 0;

  if Left < 0 then begin
    Left := ParentX + Border;
  end;

  if (Left+Width) > (ParentX + Parent.BoundsRect.w) then begin
    Left := ParentX + Parent.BoundsRect.w - Width - Border;
  end;

  if Top < ParentY then begin
    Top := ParentY + Border;
  end;

  if (Top+Height) > (ParentY + Parent.BoundsRect.h) then begin
    Top := ParentY + Parent.BoundsRect.h - Height - Border;
  end;
end;

procedure TRectangule.CentralizeAtTopWith(ARect: TSDL_Rect);
begin
  Left := ARect.x + (ARect.w div 2) - (Width  div 2);
  Top  := ARect.y - Height - 5;
end;

procedure TRectangule.MoveToBottomRightScreen;
begin
  Left := Parent.BoundsRect.w - Width - 25;
  Top  := Parent.BoundsRect.h - Height - 25;
end;

procedure TRectangule.CentralizeAtRightWith(ARect: TSDL_Rect; AFactor : Byte);
begin
  Left := ARect.x + Width * AFactor;
  Top  := ARect.y + (ARect.h div 2) - (Height div 2);
end;

procedure TRectangule.Inflate(AValue: cint);
begin
  FRect.x := FRect.x - AValue;
  FRect.y := FRect.y - AValue;
  FRect.w := FRect.w + (2 * AValue);
  FRect.h := FRect.h + (2 * AValue);
end;

procedure TRectangule.ToOriginalBounds;
begin
  FRect := FOriginalBounds;
end;

procedure TRectangule.SetOriginalBounds;
begin
  FOriginalBounds := FRect;
end;

procedure TRectangule.DoRandomMouseUp;
var
  X : cint;
  Y : cint;
begin
  X := RandomRange(0, Width);
  Y := RandomRange(0, Height);
  MouseUp(Self, [], Left+X, Top+Y);
end;

initialization
  TestMode := False;

end.

