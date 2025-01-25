{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.events.nolcl;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, ctypes, sdl2, sdl.app.events.abstract;

type

  { audio events}
  TOnAudioChannelFinished = procedure(const AChannel : cint32) of object;

  { TCustomEventHandler }

  TCustomEventHandler = class sealed(TEventHandler)
    private
      FOnAudioChannelFinished: TOnAudioChannelFinished;
      procedure SetOnAudioChannelFinished(AValue: TOnAudioChannelFinished);
      procedure UserEvent(const event: TSDL_UserEvent);
    public
      constructor Create; reintroduce;
      destructor Destroy; override;
      procedure AssignEvents;
      property OnAudioChannelFinished : TOnAudioChannelFinished read FOnAudioChannelFinished write SetOnAudioChannelFinished;
    public
      property OnMouseMotion;
      property OnMouseButtonDown;
      property OnMouseButtonUp;
      property OnKeyDown;
      property OnKeyUp;
      property OnUserEvent;
      property OnTextEditing;
      property OnTextInput;
  end;

var
  SDLEvents : TCustomEventHandler;

implementation

uses sdl.timer;

{ TCustomEventHandler }

procedure TCustomEventHandler.SetOnAudioChannelFinished(
  AValue: TOnAudioChannelFinished);
begin
  if FOnAudioChannelFinished=AValue then Exit;
  FOnAudioChannelFinished:=AValue;
end;

procedure TCustomEventHandler.UserEvent(const event: TSDL_UserEvent);

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

begin
  case event.type_ of

    SESSION_ONTIMER:
      DoOnTimer;

  end;
end;

constructor TCustomEventHandler.Create;
var
  Event : TSDL_EventType;
  SDLUserEvents : array [0..0] of TSDL_EventType =
    (SESSION_ONTIMER);
begin
  inherited Create;
  for Event in SDLUserEvents do
    if not UserEventRegistered(Event) then
      raise Exception.Create('Event not registered:'+IntToStr(Event));
  AssignEvents;
end;

destructor TCustomEventHandler.Destroy;
begin
  FOnAudioChannelFinished:=nil;
  inherited Destroy;
end;

procedure TCustomEventHandler.AssignEvents;
begin
  OnUserEvent := @UserEvent;
end;

end.

