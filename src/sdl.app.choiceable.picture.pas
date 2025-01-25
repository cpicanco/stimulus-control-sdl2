{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.choiceable.picture;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , SDL2
  , sdl.app.choiceable.rectangule
  , sdl.app.paintable.contract
  , sdl.app.events.abstract
  ;

type

  { TChoiceablePicture }

  TChoiceablePicture = class(TChoiceableRect, IPaintable)
  private
    FTexture  : PSDL_Texture;
  protected
    procedure MouseMove(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer); override;
    procedure MouseUp(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer); override;
    procedure MouseEnter(Sender: TObject); override;
    procedure MouseExit(Sender: TObject); override;
    procedure Paint; override;
  public
    constructor Create; override;
    destructor Destroy; override;
    procedure LoadFromFile(AFilename: string);
  end;


implementation

uses
  sdl2_image
  , LazFileUtils
  , sdl.app.video.methods
  //, sdl.app.output
  ;

{ TChoiceablePicture }

constructor TChoiceablePicture.Create;
begin
  inherited Create;
  IMG_Init(IMG_INIT_PNG);
end;

destructor TChoiceablePicture.Destroy;
begin
  SDL_DestroyTexture(FTexture);
  inherited Destroy;
end;

procedure TChoiceablePicture.MouseMove(Sender: TObject; Shift: TCustomShiftState; X,
  Y: Integer);
begin
  inherited MouseMove(Self, Shift, X, Y);
end;

procedure TChoiceablePicture.MouseUp(Sender: TObject; Shift: TCustomShiftState; X,
  Y: Integer);
begin
  inherited MouseUp(Self, Shift, X, Y);
end;

procedure TChoiceablePicture.MouseEnter(Sender: TObject);
begin
  inherited MouseEnter(Self);
end;

procedure TChoiceablePicture.MouseExit(Sender: TObject);
begin
  inherited MouseExit(Self);
end;

procedure TChoiceablePicture.Paint;
begin
  if Visible then begin
    SDL_RenderCopy(PSDLRenderer, FTexture, nil, @FRect);
  end else begin
    inherited Paint;
  end;
end;

procedure TChoiceablePicture.LoadFromFile(AFilename: string);
const
  IMG_EXT = '.png';
var
  Media : PAnsiChar;
begin
  CustomName := ExtractFileNameOnly(AFilename);
  Media := PAnsiChar(AFilename+IMG_EXT);
  FTexture := IMG_LoadTexture(PSDLRenderer, Media);
end;

end.


