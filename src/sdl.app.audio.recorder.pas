{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.audio.recorder;

{$mode objfpc}{$H+}

interface

uses
  Classes, sdl.app.audio.recorder.devices;

type

  { TRecorderDevice }

  TRecorderDevice = class
  private
    FOnFinished: TNotifyEvent;
    FOnStopped: TNotifyEvent;
    FRecorder : TAudioRecorderComponent;
    //FPlayback : TAudioPlaybackComponent;
    procedure DoFinished(Sender: TObject);
    procedure DoStopped(Sender: TObject);
  public
    constructor Create;
    destructor Destroy; override;
    procedure Open;
    procedure Close;
    property Recorder : TAudioRecorderComponent read FRecorder;
    //property Playback : TAudioPlaybackComponent read FPlayback;
    property OnStopped : TNotifyEvent read FOnStopped write FOnStopped;
    property OnFinished : TNotifyEvent read FOnFinished write FOnFinished;
  end;

implementation

uses
  SysUtils
  //, sdl.app.output
  ;

{ TRecorderDevice }

procedure TRecorderDevice.DoFinished(Sender: TObject);
begin
  if Assigned(OnFinished) then begin
    OnFinished(Sender);
  end;
end;

procedure TRecorderDevice.DoStopped(Sender: TObject);
begin
  if Assigned(OnStopped) then begin
    OnStopped(Sender);
  end;
end;

constructor TRecorderDevice.Create;
begin
  inherited Create;
  FRecorder := TAudioRecorderComponent.Create;
  FRecorder.Start;
  //FPlayback := TAudioPlaybackComponent.Create;
  //FPlayback.Start;
end;

destructor TRecorderDevice.Destroy;
begin
  Close;
  inherited Destroy;
end;

procedure TRecorderDevice.Open;
begin
  if not FRecorder.Opened then begin
    FRecorder.Open;
  end;

  //if not FPlayback.Opened then begin
  //  FPlayback.Open;
  //end;
  FRecorder.OnRecordingStopped := @DoStopped;
  FRecorder.OnRecordingFinished := @DoFinished;
  //FPlayback.OnPlaybackFinished := @DoFinished;
end;

procedure TRecorderDevice.Close;
begin
  FRecorder.OnRecordingFinished := nil;
  FRecorder.Close;
  FRecorder.Terminate;

  //FPlayback.OnPlaybackFinished := nil;
  //FPlayback.Close;
  //FPlayback.Terminate;
end;


end.

