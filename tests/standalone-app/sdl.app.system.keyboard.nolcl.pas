{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.system.keyboard.nolcl;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Generics.Collections, SDL2,
  sdl.app.events.abstract;

type

  TKeyDownEvents = specialize TList<TOnKeyDownEvent>;

  { TSDLSystemKeyboard }

  TSDLSystemKeyboard = class
  private
    FTextInput : string;
    FOnKeyDown : TOnKeyDownEvent;
    FOnKeyDownEvents : TKeyDownEvents;
    FOnTextInput: TOnTextInputEvent;
    //procedure ToggleActiveTextInput;
    procedure KeyDown(const event: TSDL_KeyboardEvent);
    //procedure TextInput(const event: TSDL_TextInputEvent);
    //procedure CalibrationStopped(Sender : TObject);
  public
    constructor Create;
    destructor Destroy; override;
    procedure RegisterOnKeyDown(AOnKeyDownEvent : TOnKeyDownEvent);
    property OnKeyDown : TOnKeyDownEvent read FOnKeyDown;
    property OnTextInput : TOnTextInputEvent read FOnTextInput;
  end;

implementation

uses
  ctypes,
  sdl.app.video.methods,
  sdl.app;

procedure TSDLSystemKeyboard.KeyDown(const event: TSDL_KeyboardEvent);
var
  LOnKeyDown : TOnKeyDownEvent;
begin
  case Event.keysym.sym of
    SDLK_ESCAPE: begin
      SDLApp.Terminate;
    end;

    otherwise begin
      for LOnKeyDown in FOnKeyDownEvents do begin
        LOnKeyDown(event);
      end;
    end;
  end;
end;

constructor TSDLSystemKeyboard.Create;
begin
  FOnKeyDownEvents := TKeyDownEvents.Create;
  FOnKeyDown := @KeyDown;
end;

destructor TSDLSystemKeyboard.Destroy;
begin
  FOnKeyDownEvents.Free;
  inherited Destroy;
end;

procedure TSDLSystemKeyboard.RegisterOnKeyDown(
  AOnKeyDownEvent: TOnKeyDownEvent);
begin
  if Assigned(AOnKeyDownEvent) then begin
    FOnKeyDownEvents.Add(AOnKeyDownEvent);
  end;
end;

end.

