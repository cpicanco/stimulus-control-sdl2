{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.graphics.animatedtext;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , SDL2
  , sdl.app.graphics.rectangule
  , sdl.app.graphics.text
  , sdl.app.paintable.contract
  , animation.easing
  ;

type

  { TAnimatedText }

  TAnimatedText = class(TText, IPaintable)
  private
    FIsAnimating : Boolean;
    FAnimation : TEasingAnimation;
    FAnimationRect : TSDL_Rect;
    procedure SetAnimationRect;
  protected
    procedure Paint; override;
  public
    constructor Create; override;
    destructor Destroy; override;
    procedure Animate;
    procedure Hide; override;
  end;

const
  IMG_EXT = '.png';

implementation

uses
  sdl.colors,
  sdl.app.video.methods,
  //sdl.app.output,
  sdl.app.testmode;

{ TAnimatedText }

constructor TAnimatedText.Create;
begin
  inherited Create;
  FIsAnimating := False;
  FAnimation := TEasingAnimation.Create;
end;

destructor TAnimatedText.Destroy;
begin
  FAnimation.Free;
  inherited Destroy;
end;

procedure TAnimatedText.SetAnimationRect;
begin
  FAnimationRect.x := FRect.x;
  FAnimationRect.y := FRect.y + FRect.h;
  FAnimationRect.w := FRect.w;
  FAnimationRect.h := FRect.h div 10;
end;

procedure TAnimatedText.Paint;
begin
  inherited Paint;
  if Visible then begin
    if FIsAnimating then begin
      SDL_SetRenderDrawBlendMode(PSDLRenderer, SDL_BLENDMODE_BLEND);
      FAnimation.StepIt;

      with clLightRedShaded1 do begin
        SDL_SetRenderDrawColor(PSDLRenderer, r, g, b, FAnimation.Value);
      end;

      SDL_RenderFillRect(PSDLRenderer, @FAnimationRect);
    end;
  end else begin
    if TestMode then begin
      with clGray do
        SDL_SetRenderDrawColor(PSDLRenderer, r, g, b, a);
      SDL_RenderFillRect(PSDLRenderer, @FRect);
    end;
  end;
end;

procedure TAnimatedText.Animate;
begin
  SetAnimationRect;
  FAnimation.Reset;
  FIsAnimating := True;
end;

procedure TAnimatedText.Hide;
begin
  inherited Hide;
  FIsAnimating := False;
  FAnimation.Reset;
end;

end.

