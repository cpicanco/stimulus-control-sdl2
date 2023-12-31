{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.audio.recorder;

{$mode objfpc}{$H+}

interface

uses
  Classes, fgl
  , sdl.app.audio.recorder.devices
  , sdl.app.graphics.toggle;

type

  TButtonContainer = specialize TFPGList<TToggleButton>;

  { TRecorderDevice }

  TRecorderDevice = class
  private
    FRecorder : TAudioRecorderComponent;
    FPlayback : TAudioPlaybackComponent;
    FContainer: TButtonContainer;
    procedure DoFinished(Sender: TObject);
  public
    constructor Create;
    destructor Destroy; override;
    procedure Open;
    procedure Close;
    procedure Append(AButton: TToggleButton);
    procedure Clear;
    procedure RadioToggle(AButton: TToggleButton);
    property Recorder : TAudioRecorderComponent read FRecorder;
    property Playback : TAudioPlaybackComponent read FPlayback;
  end;

implementation

uses SysUtils, session.pool
  , sdl.app.output, sdl.app.stimulus, sdl.app.stimulus.contract;

{ TRecorderDevice }

procedure TRecorderDevice.DoFinished(Sender: TObject);
var
  LButton : TToggleButton = nil;
  LStimulus : IStimulus;
begin
  if Assigned(Sender) then begin
    if Sender = FRecorder then begin
      if Assigned(FRecorder.Starter) then begin
        LButton := TToggleButton(FRecorder.Starter);
        LStimulus := LButton.Owner as IStimulus;
        LStimulus.DoResponse(True);
      end;
    end;

    if Sender = FPlayback then begin
      if Assigned(FPlayback.Starter) then begin
        LButton := TToggleButton(FPlayback.Starter);
      end;
    end;

    if Assigned(LButton) then begin
      RadioToggle(LButton);
    end;
  end;
end;

constructor TRecorderDevice.Create;
begin
  inherited Create;
  FContainer := TButtonContainer.Create;
  FRecorder := TAudioRecorderComponent.Create;
  FRecorder.Start;
  FPlayback := TAudioPlaybackComponent.Create;
  FPlayback.Start;
end;

destructor TRecorderDevice.Destroy;
begin
  Close;
  FContainer.Free;
  inherited Destroy;
end;

procedure TRecorderDevice.Open;
begin
  if not FRecorder.Opened then begin
    FRecorder.Open;
  end;

  if not FPlayback.Opened then begin
    FPlayback.Open;
  end;
  FRecorder.OnRecordingFinished := @DoFinished;
  FPlayback.OnPlaybackFinished := @DoFinished;
end;

procedure TRecorderDevice.Close; // todo: fix memory leak here
begin
  FRecorder.OnRecordingFinished := nil;
  FRecorder.Close;
  FRecorder.Terminate;

  FPlayback.OnPlaybackFinished := nil;
  FPlayback.Close;
  FPlayback.Terminate;
end;

procedure TRecorderDevice.Append(AButton: TToggleButton);
begin
  FContainer.Add(AButton);
end;

procedure TRecorderDevice.Clear;
begin
  FContainer.Clear;
end;

procedure TRecorderDevice.RadioToggle(AButton: TToggleButton);
var
  LButton : TToggleButton;
begin
  for LButton in FContainer do begin
    if LButton = AButton then begin
      LButton.Enabled := False;
    end else begin
      LButton.Enabled := True;
    end;
  end;
end;


end.

