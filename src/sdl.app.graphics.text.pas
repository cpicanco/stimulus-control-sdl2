{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.graphics.text;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , ctypes
  , sdl2
  , sdl2_ttf
  , sdl.app.graphics.rectangule
  ;

type

  { TText }

  TText = class(TRectangule)
    private
      FFont : PTTF_Font;
      FFontName: string;
      FSDLTexture : PSDL_Texture;
      FWrapped: Boolean;
      FWrappedWidth: cint32;
      function GetFontSize: cint;
      function GetFontStyle: cint;
      procedure SetFontName(AValue: string);
      procedure SetFontSize(AValue: cint);
      procedure SetFontStyle(AValue: cint);
      procedure SetWrapped(AValue: Boolean);
      procedure SetWrappedWidth(AValue: integer);
    protected
      procedure Paint; override;
    public
      constructor Create; override;
      procedure Load(AString : string);
      procedure LoadFromFile(AFilename: string);
      destructor Destroy; override;
      procedure Clear;
      //procedure Show;
      //procedure Hide;
      //property Visible : Boolean read FVisible write FVisible;
      property FontName : string read FFontName write SetFontName;
      property FontSize : cint read GetFontSize write SetFontSize;
      property FontStyle: cint read GetFontStyle write SetFontStyle;
      property Wrapped : Boolean read FWrapped write SetWrapped;
      property WrappedWidth : integer read FWrappedWidth write SetWrappedWidth;
  end;

implementation

uses
  sdl.app.video.methods,
  sdl.app.text,
  sdl.colors,
  session.strutils;

{ TText }

procedure TText.SetWrapped(AValue: Boolean);
begin
  if FWrapped = AValue then Exit;
  FWrapped := AValue;
end;

procedure TText.SetFontName(AValue: string);
begin
  if FFontName = AValue then Exit;
  FFontName := AValue;
  FFont := SDLText.Get(FontName).Font;
end;

function TText.GetFontSize: cint;
begin
  if Assigned(FFont) then
    Result := TTF_FontHeight(FFont);
end;

function TText.GetFontStyle: cint;
begin
  Result := TTF_GetFontStyle(FFont);
end;

procedure TText.SetFontSize(AValue: cint);
begin
  if GetFontSize = AValue then Exit;
  TTF_SetFontSize(FFont, AValue);
end;

procedure TText.SetFontStyle(AValue: cint);
begin
  if GetFontStyle = AValue then Exit;
  TTF_SetFontStyle(FFont, AValue);
end;

procedure TText.SetWrappedWidth(AValue: integer);
begin
  if FWrappedWidth = AValue then Exit;
  FWrappedWidth := AValue;
end;

procedure TText.Paint;
begin
  inherited Paint;
  if Visible then begin
    SDL_RenderCopy(PSDLRenderer, FSDLTexture, nil, @FRect);
  end;
end;

constructor TText.Create;
begin
  inherited Create;
  //FWrapped := False;
  //FWrappedWidth := 800;
  //FFontName := '';
  //FFont := nil;
  //FSDLTexture := nil;
  //Visible := False;
end;

procedure TText.Load(AString: string);
//const
//  WrapOnNewLine : cuint32 = 0;
var
  PSDLSurface : PSDL_Surface;
begin
  if AString.IsEmpty then begin
    raise Exception.Create('TText.Load cannot load empty strings');
  end;
  //PSDLSurface := TTF_RenderUTF8_LCD(
  if Wrapped then begin
    PSDLSurface := TTF_RenderUTF8_Blended_Wrapped(
      FFont, PAnsiChar(AString), clFontColor, WrappedWidth);
  end else begin
    PSDLSurface := TTF_RenderUTF8_Blended(
      FFont, PAnsiChar(AString), clFontColor);
  end;
  FRect := PSDLSurface^.clip_rect;
  FSDLTexture := SDL_CreateTextureFromSurface(PSDLRenderer, PSDLSurface);
  SDL_FreeSurface(PSDLSurface);
end;

procedure TText.LoadFromFile(AFilename: string);
var
  LFile : TStringList;
  LText : string;
const
  TXT_EXT = '.txt';
begin
  LFile := TStringList.Create;
  try
    LFile.LoadFromFile(AsInstruction(AFilename)+TXT_EXT);
    LText := LFile[0];
  finally
    LFile.Clear;
    LFile.Free;
  end;
  Load(LText);
end;

destructor TText.Destroy;
begin
  SDL_DestroyTexture(FSDLTexture);
  inherited Destroy;
end;

procedure TText.Clear;
begin
  SDL_DestroyTexture(FSDLTexture);
end;

end.

