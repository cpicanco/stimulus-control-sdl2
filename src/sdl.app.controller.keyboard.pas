{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.controller.keyboard;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils, SDL2,
  sdl.app.events.abstract,
  sdl.app.navigator.contract,
  sdl.app.controller;

type

  { TSDLKeyboard }

  TSDLKeyboard = class(TController)
  private
    procedure KeyDown(const event: TSDL_KeyboardEvent);
  public
    constructor Create; override;
  end;

implementation

uses sdl.app.events.custom;

{ TSDLKeyboard }

procedure TSDLKeyboard.KeyDown(const event: TSDL_KeyboardEvent);
begin
  if FNavigator = nil then Exit;

  case Event.keysym.sym of

    SDLK_LEFT: begin
      FNavigator.GoLeft;
    end;

    SDLK_RIGHT : begin
      FNavigator.GoRight;
    end;

    SDLK_UP : begin
      FNavigator.GoTop;
    end;

    SDLK_DOWN: begin
      FNavigator.GoBottom;
    end;

    SDLK_RETURN, SDLK_RETURN2: begin
      FNavigator.ConfirmSelection;
    end;

    SDLK_BACKSPACE: begin
      //FNavigator.Unselect;
    end;
  end;
end;

constructor TSDLKeyboard.Create;
begin
  inherited Create;
  SDLEvents.Keyboard.RegisterOnKeyDown(@KeyDown);
end;

end.

