{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.graphics.picture;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , SDL2
  , sdl.app.graphics.rectangule
  , sdl.app.paintable.contract
  , sdl.app.events.abstract
  ;

type

  { TPicture }

  TPicture = class(TRectangule, IPaintable)
  private
    FSibling: TRectangule;
    FStimulus: TObject;
    procedure SetSibling(AValue: TRectangule);
    procedure SetStimulus(AValue: TObject);
  protected
    FTexture  : PSDL_Texture;
    //procedure MouseMove(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer); override;
    //procedure MouseUp(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer); override;
    //procedure MouseEnter(Sender: TObject); override;
    //procedure MouseExit(Sender: TObject); override;
    procedure Paint; override;
  public
    constructor Create; override;
    destructor Destroy; override;
    procedure LoadFromFile(AFilename: string); virtual;
    property Sibling : TRectangule read FSibling write SetSibling;
  end;

const
  IMG_EXT = '.png';

implementation

uses
  sdl2_image
  //, sdl.colors
  , sdl.app.video.methods
  //, sdl.app.output
  ;

{ TPicture }

constructor TPicture.Create;
begin
  inherited Create;
  FSibling := nil;
end;

destructor TPicture.Destroy;
begin
  SDL_DestroyTexture(FTexture);
  inherited Destroy;
end;

procedure TPicture.SetSibling(AValue: TRectangule);
begin
  if FSibling=AValue then Exit;
  FSibling:=AValue;
end;

procedure TPicture.SetStimulus(AValue: TObject);
begin
  if FStimulus = AValue then Exit;
  FStimulus := AValue;
end;

//procedure TPicture.MouseMove(Sender: TObject; Shift: TCustomShiftState; X,
//  Y: Integer);
//begin
//  if Visible then
//    inherited MouseMove(Self, Shift, X, Y);
//end;

//procedure TPicture.MouseUp(Sender: TObject; Shift: TCustomShiftState; X,
//  Y: Integer);
//begin
//  if Visible then
//    inherited MouseUp(Self, Shift, X, Y);
//end;

//procedure TPicture.MouseEnter(Sender: TObject);
//begin
//  if Visible then begin
//    inherited MouseEnter(Self);
//
//  end;
//end;

//procedure TPicture.MouseExit(Sender: TObject);
//begin
//  if Visible then begin
//    inherited MouseExit(Self);
//
//  end;
//end;

procedure TPicture.Paint;
begin
  inherited Paint;
  if Visible then begin
    SDL_RenderCopy(PSDLRenderer, FTexture, nil, @FRect);
  end;
end;

procedure TPicture.LoadFromFile(AFilename: string);
var
  Media : PAnsiChar;
begin
  Media := PAnsiChar(AFilename+IMG_EXT);
  FTexture := IMG_LoadTexture(PSDLRenderer, Media);
end;

end.

