{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit eye.tracker.pupil;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , SDL2
  , Pupil.Types
  , Pupil.Client
  , eye.tracker.types
  , eye.tracker.client
  ;

type

  { TPupilEyeTracker }

  TPupilEyeTracker = class sealed (TEyeTrackerClient)
    private
      FCurrentData : TGazeOnSurface;
      FPupilClient : TPupilClient;
      FGazeOnScreenEvent : TGazeOnScreenEvent;
      procedure GazeOnSurface(Sender : TObject; AGazeOnSurface: TGazeOnSurface);
    protected
      function TrackerClassName : string; override;
      function GetGazeOnScreenEvent: TGazeOnScreenEvent; override;
      procedure SetGazeOnScreenEvent(AValue: TGazeOnScreenEvent); override;
      procedure SetOnCalibrationSuccessful(AValue: TNotifyEvent); override;
      procedure SetOnCalibrationFailed(AValue: TNotifyEvent); override;
      procedure StartRecording; override;
      procedure StopRecording; override;
      procedure StartCalibration; override;
      procedure StopCalibration; override;
      procedure CalibrationSuccessful; override;
    public
      constructor Create;
      destructor Destroy; override;
      function CurrentGazes : TNormalizedGazes; override;
  end;

implementation

{ TPupilEyeTracker }

procedure TPupilEyeTracker.SetGazeOnScreenEvent(AValue: TGazeOnScreenEvent);
begin
  if FGazeOnScreenEvent = AValue then Exit;
  FGazeOnScreenEvent := AValue;
end;

procedure TPupilEyeTracker.SetOnCalibrationSuccessful(
  AValue: TNotifyEvent);
begin
  with FPupilClient do begin
    if OnCalibrationSuccessful = AValue then Exit;
    OnCalibrationSuccessful := AValue;
  end;
end;

procedure TPupilEyeTracker.SetOnCalibrationFailed(
  AValue: TNotifyEvent);
begin
  with FPupilClient do begin
    if OnCalibrationFailed = AValue then Exit;
    OnCalibrationFailed := AValue;
  end;
end;

procedure TPupilEyeTracker.StartRecording;
begin
  FPupilClient.Request(REQ_SHOULD_START_RECORDING);
end;

procedure TPupilEyeTracker.StopRecording;
begin
  FPupilClient.Request(REQ_SHOULD_STOP_RECORDING, True);
end;

procedure TPupilEyeTracker.StartCalibration;
begin
  FPupilClient.Request(REQ_SHOULD_START_CALIBRATION, True);
end;

procedure TPupilEyeTracker.StopCalibration;
begin
  FPupilClient.Request(REQ_SHOULD_STOP_CALIBRATION);
end;

procedure TPupilEyeTracker.CalibrationSuccessful;
begin

end;

constructor TPupilEyeTracker.Create;
begin
  FPupilClient := TPupilClient.Create;
  FPupilClient.Start;
  FPupilClient.StartSubscriber;
  FPupilClient.OnGazeOnSurface := @GazeOnSurface;
end;

destructor TPupilEyeTracker.Destroy;
begin
  FPupilClient.Close;
  //FPupilClient.Free;
  inherited Destroy;
end;

function TPupilEyeTracker.CurrentGazes: TNormalizedGazes;
begin
  Result := TNormalizedGazes(FCurrentData.Gazes);
end;

procedure TPupilEyeTracker.GazeOnSurface(Sender : TObject;
  AGazeOnSurface: TGazeOnSurface);
var
  event : TSDL_Event;
begin
  FCurrentData := AGazeOnSurface;
  event.type_ := EYE_TRACKER_GAZE_EVENT;
  event.user.data1 := Pointer(Self);
  SDL_PushEvent(@event);
end;

function TPupilEyeTracker.TrackerClassName: string;
begin
  Result := Self.ClassName;
end;

function TPupilEyeTracker.GetGazeOnScreenEvent: TGazeOnScreenEvent;
begin
  Result := FGazeOnScreenEvent;
end;

end.

