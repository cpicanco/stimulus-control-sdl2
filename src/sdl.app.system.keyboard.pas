{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.system.keyboard;

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
    procedure ToggleActiveTextInput;
    procedure KeyDown(const event: TSDL_KeyboardEvent);
    procedure TextInput(const event: TSDL_TextInputEvent);
    procedure CalibrationStopped(Sender : TObject);
  public
    constructor Create;
    destructor Destroy; override;
    procedure RegisterOnKeyDown(AOnKeyDownEvent : TOnKeyDownEvent);
    property OnKeyDown : TOnKeyDownEvent read FOnKeyDown;
    property OnTextInput : TOnTextInputEvent read FOnTextInput;
  end;

implementation

uses
  sdl.app.video.methods,
  ctypes,
  eye.tracker,
  session.loggers.writerow.timestamp,
  devices.RS232i,
  sdl.app,
  sdl.app.trials.factory,
  sdl.app.controller.manager,
  Forms.speechvalidation;

{ TSDLKeyboard }

procedure TSDLSystemKeyboard.ToggleActiveTextInput;
  procedure Activate;
  begin
    FormSpeechValidationQueue.Show;
    RaiseWindow;
  end;

  procedure Deactivate;
  begin
    FormSpeechValidationQueue.Hide;
  end;

begin
  if SDL_IsTextInputActive and
     (not FormSpeechValidationQueue.Visible) then begin
    Activate;
  end;

  if SDL_IsTextInputActive then begin
    SDL_StopTextInput;
    Deactivate;
  end else begin
    SDL_StartTextInput;
    Activate;
  end;
end;

procedure TSDLSystemKeyboard.KeyDown(const event: TSDL_KeyboardEvent);
var
  LOnKeyDown : TOnKeyDownEvent;
  LKeyboardState: pcuint8 = nil;
  LKey : Char;
begin

  case Event.keysym.sym of
    SDLK_ESCAPE: begin
      SDLApp.Terminate;
    end;

    //SDLK_BACKSPACE: begin
    //  SetLength(FTextInput, Length(FTextInput) - 1);
    //  with FormSpeechValidationQueue do begin
    //    EditSpeech.Text := FTextInput;
    //  end;
    //end;

    //SDLK_RETURN: begin
    //  LKeyboardState := SDL_GetKeyboardState(nil);
    //  if GetKeyState(SDL_SCANCODE_LCTRL, LKeyboardState) then begin
    //    ToggleActiveTextInput;
    //  end else begin
    //    FTextInput := '';
    //    LKey := #13;
    //    with FormSpeechValidationQueue do begin
    //      EditSpeechKeyPress(Self, LKey);
    //      EditSpeech.Text := FTextInput;
    //    end;
    //  end;
    //end;

    SDLK_c: begin
      LKeyboardState := SDL_GetKeyboardState(nil);
      if GetKeyState(SDL_SCANCODE_LCTRL, LKeyboardState) then begin
        if EyeTracker <> nil then begin
          EyeTracker.SetOnCalibrationFailed(@CalibrationStopped);
          EyeTracker.SetOnCalibrationSuccessful(@CalibrationStopped);
          EyeTracker.StartCalibration;
          Timestamp(EyeTracker.TrackerClassName+'.StartCalibration');
        end;
      end;
    end;

    SDLK_p: begin
      LKeyboardState := SDL_GetKeyboardState(nil);
      if GetKeyState(SDL_SCANCODE_LCTRL, LKeyboardState) then begin
        Controllers.Reboot;
      end;
    end;

    //SDLK_s: begin
    //  LKeyboardState := SDL_GetKeyboardState(nil);
    //  if GetKeyState(SDL_SCANCODE_LCTRL, LKeyboardState) then begin
    //    if EyeTracker <> nil then begin
    //      EyeTracker.CalibrationSuccessful;
    //    end;
    //  end;
    //end;

    //SDLK_n: begin
    //  LKeyboardState := SDL_GetKeyboardState(nil);
    //  if GetKeyState(SDL_SCANCODE_LCTRL, LKeyboardState) then begin
    //    TTrialFactory.CurrentTrial.DoExpectedResponse;
    //  end;
    //end;

    SDLK_SPACE : begin
      LKeyboardState := SDL_GetKeyboardState(nil);
      if GetKeyState(SDL_SCANCODE_LCTRL, LKeyboardState) then begin
        TTrialFactory.CurrentTrial.Reboot;
      end;
    end;

    SDLK_KP_ENTER: begin
      LKeyboardState := SDL_GetKeyboardState(nil);
      if GetKeyState(SDL_SCANCODE_LCTRL, LKeyboardState) then begin
        if Assigned(RS232) then begin
          RS232.Dispenser;
        end;
      end;
    end;

    otherwise begin
      for LOnKeyDown in FOnKeyDownEvents do begin
        LOnKeyDown(event);
      end;
    end;
  end;
end;

procedure TSDLSystemKeyboard.TextInput(const event: TSDL_TextInputEvent);
begin
  FTextInput += Event.text;
  FormSpeechValidationQueue.EditSpeech.Text := FTextInput;
end;

procedure TSDLSystemKeyboard.CalibrationStopped(Sender: TObject);
begin
  EyeTracker.StopCalibration;
  Timestamp(EyeTracker.TrackerClassName+'.StopCalibration');
  RaiseWindow;
end;

constructor TSDLSystemKeyboard.Create;
begin
  FOnKeyDownEvents := TKeyDownEvents.Create;
  FOnKeyDown:=@KeyDown;
  FOnTextInput:=@TextInput;
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

