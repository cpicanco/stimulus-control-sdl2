{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.events.custom;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, ctypes
  , sdl2
  , sdl.app.events.abstract
  , sdl.app.system.keyboard
  , eye.tracker.types;

type

  { audio events}
  TOnAudioChannelFinished = procedure(const AChannel : cint32) of object;

  { TCustomEventHandler }

  TCustomEventHandler = class sealed(TEventHandler)
    private
      FKeyboard: TSDLSystemKeyboard;
      FOnAudioChannelFinished: TOnAudioChannelFinished;
      function GetOnGazeOnScreen: TGazeOnScreenEvent;
      procedure SetOnAudioChannelFinished(AValue: TOnAudioChannelFinished);
      procedure SetOnGazeOnScreen(AValue: TGazeOnScreenEvent);
      procedure UserEvent(const event: TSDL_UserEvent);
    public
      constructor Create; reintroduce;
      destructor Destroy; override;
      procedure AssignEvents;
      property OnAudioChannelFinished : TOnAudioChannelFinished read FOnAudioChannelFinished write SetOnAudioChannelFinished;
      property OnGazeOnScreen : TGazeOnScreenEvent read GetOnGazeOnScreen write SetOnGazeOnScreen;
      property Keyboard : TSDLSystemKeyboard read FKeyboard write FKeyboard;
    public
      property OnMouseMotion;
      property OnMouseButtonDown;
      property OnMouseButtonUp;
      property OnKeyDown;
      property OnKeyUp;
      property OnUserEvent;
      property OnTextEditing;
      property OnTextInput;
      property OnControllerAxisMotion;
      property OnControllerButtonDown;
      property OnControllerButtonUp;
      property OnControllerTouchPadMotion;
      property OnControllerSensorUpdate;
  end;

var
  SDLEvents : TCustomEventHandler;

implementation

uses
  sdl.app.trials,
  sdl.timer,
  sdl.app.audio,
  sdl.app.audio.recorder.devices,
  eye.tracker;

{ TCustomEventHandler }

procedure TCustomEventHandler.SetOnAudioChannelFinished(
  AValue: TOnAudioChannelFinished);
begin
  if FOnAudioChannelFinished=AValue then Exit;
  FOnAudioChannelFinished:=AValue;
end;

function TCustomEventHandler.GetOnGazeOnScreen: TGazeOnScreenEvent;
begin
  Result := nil;
  if Assigned(EyeTracker) then begin
    Result := EyeTracker.GetGazeOnScreenEvent;
  end;
end;

procedure TCustomEventHandler.SetOnGazeOnScreen(AValue: TGazeOnScreenEvent);
begin
  if Assigned(EyeTracker) then begin
    if OnGazeOnScreen = AValue then Exit;
    EyeTracker.SetGazeOnScreenEvent(AValue);
  end;
end;

procedure TCustomEventHandler.UserEvent(const event: TSDL_UserEvent);

  procedure DoOnEndSound;
  begin
    if Assigned(OnAudioChannelFinished) then begin
      OnAudioChannelFinished(event.code);
    end;
  end;

  procedure DoOnEndTrial;
  var
    LOnTrialEnd : TNotifyEvent;
    LTrial: TTrial;
  begin
    LTrial := TTrial(event.data1);
    if Assigned(LTrial) then begin
      LOnTrialEnd := LTrial.OnTrialEnd;
      if Assigned(LOnTrialEnd) then begin
        LOnTrialEnd(LTrial);
      end;
    end;
  end;

  procedure DoOnTimer;
  var
    LTimer : TSDLTimer;
  begin
    LTimer := TSDLTimer(event.data1);
    if Assigned(LTimer) then begin
      if Assigned(LTimer.OnTimer) then begin
        LTimer.OnTimer(LTimer);
      end;
    end;
  end;

  procedure DoOnRecordingFinished;
  var
    LRecorder : TAudioRecorderComponent;
  begin
    LRecorder := TAudioRecorderComponent(event.data1);
    if Assigned(LRecorder) then begin
      if Assigned(LRecorder.OnRecordingFinished) then begin
        LRecorder.OnRecordingFinished(LRecorder);
      end;
    end;
  end;

  procedure DoOnRecordingStopped;
  var
    LRecorder : TAudioRecorderComponent;
  begin
    LRecorder := TAudioRecorderComponent(event.data1);
    if Assigned(LRecorder) then begin
      if Assigned(LRecorder.OnRecordingStopped) then begin
        LRecorder.OnRecordingStopped(LRecorder);
      end;
    end;
  end;

begin
  case event.type_ of
    SESSION_TRIALEND:
      DoOnEndTrial;

    SESSION_ONTIMER:
      DoOnTimer;

    SESSION_CHUNK_STOPPED:
      DoOnEndSound;

    EYE_TRACKER_GAZE_EVENT: begin
      OnGazeOnScreen(EyeTracker.CurrentGazes);
    end;

    SESSION_RECORDING_FINISHED:
      DoOnRecordingFinished;

    SESSION_RECORDING_STOPPED:
      DoOnRecordingStopped;
  end;
end;

constructor TCustomEventHandler.Create;
var
  Event : TSDL_EventType;
  SDLUserEvents : array [0..SDL_USEREVENTSTOREGISTER] of TSDL_EventType = (
    SESSION_TRIALEND,
    SESSION_ONTIMER,
    SESSION_CHUNK_STOPPED,
    EYE_TRACKER_GAZE_EVENT,
    SESSION_RECORDING_FINISHED,
    SESSION_RECORDING_STOPPED);
begin
  inherited Create;
  FKeyboard := TSDLSystemKeyboard.Create;
  OnKeyDown := FKeyboard.OnKeyDown;
  OnTextInput := FKeyboard.OnTextInput;
  for Event in SDLUserEvents do
    if not UserEventRegistered(Event) then
      raise Exception.Create('Event not registered:'+IntToStr(Event));
  AssignEvents;
end;

destructor TCustomEventHandler.Destroy;
begin
  FKeyboard.Free;
  FOnAudioChannelFinished:=nil;
  inherited Destroy;
end;

procedure TCustomEventHandler.AssignEvents;
begin
  OnUserEvent := @UserEvent;
end;

end.

