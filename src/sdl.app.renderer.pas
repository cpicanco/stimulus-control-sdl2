{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.renderer;

{$mode ObjFPC}{$H+}

interface

uses sdl2;

//type
//
//  //Assigned to TTrial.Parent;
//  TSDLRenderer = class
//    class var Color : TSDL_Color;
//    class procedure Paint;
//  end;

//procedure Render;
procedure RenderOptimized;

implementation

uses
  //sdl.app.renderer.testmode,
  sdl.app.renderer.types,
  sdl.app.renderer.validation,
  sdl.app.video.methods,
  sdl.app.trials.factory,
  sdl.app.graphics.debug,
  sdl.app.markers,
  sdl.colors;


//procedure Render;
//begin
//  SDL_SetRenderDrawColor(PSDLRenderer,
//    clBackgroud.r, clBackgroud.g, clBackgroud.b, clBackgroud.a);
//  SDL_RenderClear(PSDLRenderer);
//
//  if Assigned(TTrialFactory.CurrentTrial) then begin
//    TTrialFactory.CurrentTrial.AsIPaintable.Paint;
//  end;
//
//  SDL_RenderPresent(PSDLRenderer);
//  SDL_Delay(DELTA_TIME);
//end;

procedure RenderOptimized;
begin
  if GPaintingInvalidated then begin
    GPaintingInvalidated := False;

    SDL_SetRenderDrawColor(PSDLRenderer,
      clBackground.r, clBackground.g, clBackground.b, clBackground.a);
    SDL_RenderClear(PSDLRenderer);

    if Assigned(Markers) then begin
      Markers.Paint;
    end;

    TTrialFactory.CurrentTrial.AsIPaintable.Paint;

    SDL_RenderPresent(PSDLRenderer);
    //SaveFrame;
  end;
  SDL_Delay(DELTA_TIME);
end;


end.



