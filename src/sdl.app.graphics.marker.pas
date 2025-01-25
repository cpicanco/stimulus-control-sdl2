{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.graphics.marker;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , ctypes
  , SDL2
  , sdl.app.graphics.rectangule
  , sdl.app.paintable.contract
  ;

type

  { TMarker }

  TMarker = class(IPaintable)
  private
    FInvalidated : Boolean;
    FParent: TSDL_Rect;
    FRect : TSDL_Rect;
    function GetHeight: cint;
    function GetLeft: cint;
    function GetTop: cint;
    function GetWidth: cint;
    procedure SetHeight(AValue: cint);
    procedure SetLeft(AValue: cint);
    procedure SetTop(AValue: cint);
    procedure SetWidth(AValue: cint);
  protected
    FTexture  : PSDL_Texture;
  public
    constructor Create;
    destructor Destroy; override;
    function Invalidated : Boolean;
    procedure LoadFromFile(AFilename: string); virtual;
    procedure Invalidate;
    procedure Validate;
    procedure ToTopLeft;
    procedure ToTopRight;
    procedure ToBottomLeft;
    procedure ToBottomRight;
    procedure Paint;
    property Parent : TSDL_Rect read FParent write FParent;
    property Left   : cint read GetLeft write SetLeft;
    property Height : cint read GetHeight write SetHeight;
    property Top    : cint read GetTop write SetTop;
    property Width  : cint read GetWidth write SetWidth;
  end;

implementation

uses
  sdl2_image
  //, sdl.colors
  , sdl.app.video.methods
  //, sdl.app.output
  ;

{ TMarker }

destructor TMarker.Destroy;
begin
  SDL_DestroyTexture(FTexture);
  inherited Destroy;
end;

function TMarker.Invalidated: Boolean;
begin
  Result := FInvalidated;
end;

function TMarker.GetHeight: cint;
begin
  Result := FRect.h;
end;

function TMarker.GetLeft: cint;
begin
  Result := FRect.x;
end;

function TMarker.GetTop: cint;
begin
  Result := FRect.y;
end;

function TMarker.GetWidth: cint;
begin
  Result := FRect.w;
end;

procedure TMarker.SetHeight(AValue: cint);
begin
  if FRect.h=AValue then Exit;
  FRect.h := AValue;
end;

procedure TMarker.SetLeft(AValue: cint);
begin
  if FRect.x=AValue then Exit;
  FRect.x:=AValue;
end;

procedure TMarker.SetTop(AValue: cint);
begin
  if FRect.y=AValue then Exit;
  FRect.y:=AValue;
end;

procedure TMarker.SetWidth(AValue: cint);
begin
  if FRect.w=AValue then Exit;
  FRect.w:=AValue;
end;

constructor TMarker.Create;
begin
  FTexture := nil;
end;

procedure TMarker.Paint;
begin
  SDL_RenderCopy(PSDLRenderer, FTexture, nil, @FRect);
end;

procedure TMarker.LoadFromFile(AFilename: string);
const
  IMG_EXT = '.png';
var
  Media : PAnsiChar;
begin
  Media := PAnsiChar(AFilename+IMG_EXT);
  FTexture := IMG_LoadTexture(PSDLRenderer, Media);
end;

procedure TMarker.Invalidate;
begin
  FInvalidated := True;
end;

procedure TMarker.Validate;
begin
  FInvalidated := False;
end;

procedure TMarker.ToTopLeft;
begin
  Top := 0;
  Left := 0;
end;

procedure TMarker.ToTopRight;
begin
  Top := FParent.y;
  Left := FParent.w - Width;
end;

procedure TMarker.ToBottomLeft;
begin
  Top := FParent.h - Height;
  Left := 0;
end;

procedure TMarker.ToBottomRight;
begin
  Top := FParent.h - Height;
  Left := FParent.w - Width;
end;

end.

