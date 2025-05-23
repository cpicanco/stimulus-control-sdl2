{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.task;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, SDL2;

type
  TOnKeyDownEvent = procedure(const event: TSDL_KeyboardEvent) of object;

  { TKeyboardTask }

  TKeyboardTask = class
  private
    FOnKeyDown: TOnKeyDownEvent;
    procedure KeyDown(const event: TSDL_KeyboardEvent);
  public
    constructor Create;
    class procedure Render;
    class procedure SetupMonitor(AMonitor : TSDL_Rect);
    property OnKeyDown: TOnKeyDownEvent read FOnKeyDown write FOnKeyDown;
  end;

implementation

uses
  sdl.colors,
  sdl.app.video.methods,
  sdl.app.renderer.validation;

{ TKeyboardTask }

var
  LRect : TSDL_Rect = (x: 0; y: 0; w: 200; h: 200);
  LMonitor : TSDL_Rect = (x: 0; y: 0; w: 800; h: 600);

procedure TKeyboardTask.KeyDown(const event: TSDL_KeyboardEvent);
begin
  case Event.keysym.sym of
    SDLK_SPACE: begin
      GPaintingInvalidated := True;
    end;
  end;
end;

class procedure TKeyboardTask.Render;
begin
  with clBlack do begin
    SDL_SetRenderDrawColor(PSDLRenderer, r, g, b, a);
  end;
  SDL_RenderClear(PSDLRenderer);

  LRect.x := Random(LMonitor.w-LRect.w);
  LRect.y := Random(LMonitor.h-LRect.h);

  with clGray do begin
    SDL_SetRenderDrawColor(PSDLRenderer, r, g, b, a);
  end;
  SDL_RenderFillRect(PSDLRenderer, @LRect);
end;

class procedure TKeyboardTask.SetupMonitor(AMonitor: TSDL_Rect);
begin
  LMonitor := AMonitor;
end;

constructor TKeyboardTask.Create;
begin
  OnKeyDown := @KeyDown;
end;

end.

