{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.controller.factory;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils,
  sdl.app.controller.types,
  sdl.app.controller;

type

  { TControllerFactory }

  TControllerFactory = class
    class function New(ACode : TControllerCode) : TController;
  end;

implementation

uses
  SDL2,
  sdl.app.controller.mouse,
  sdl.app.controller.keyboard,
  sdl.app.controller.wii,
  sdl.app.controller.ps4;

{ TControllerFactory }

class function TControllerFactory.New(ACode: TControllerCode): TController;
begin
  SDL_JoystickUpdate;
  if SDL_NumJoysticks < 1 then ACode := gcMouse;
  case ACode of
    gcMouse: begin
      Result := TSDLMouseController.Create;
    end;

    gcKeyboard: begin
      Result := TSDLKeyboard.Create;
    end;

    gcWii: begin
      Result := TSDLWiiRemote.Create;
    end;

    gcPS4: begin
      Result := TSDLPS4Controller.Create;
    end;
  end;
  Result.ControllerCode := ACode;
end;

end.


