{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit eye.tracker.gazepoint;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , SDL2
  , opengaze.types
  , eye.tracker.types
  , eye.tracker.client
  ;

type

  { TGazepointEyeTracker }

  TGazepointEyeTracker = class sealed (TEyeTrackerClient)
    private
      FReceiveGazeEvent : TGazeOnScreenEvent;
      procedure ReceiveNormalizedGaze(Sender : TObject; RawTag: TRawTag);
    protected
      function IsFake : Boolean; override;
      function GetGazeOnScreenEvent: TGazeOnScreenEvent; override;
      function TrackerClassName : string; override;
      procedure SetGazeOnScreenEvent(AValue: TGazeOnScreenEvent); override;
      procedure SetOnCalibrationSuccessful(AValue: TNotifyEvent); override;
      procedure SetOnCalibrationFailed(AValue: TNotifyEvent); override;
      procedure StartRecording; override;
      procedure StopRecording; override;
      procedure StartCalibration; override;
      procedure StopCalibration; override;
      procedure CalibrationSuccessful; override;
      procedure SetDataFilename(AFilename : string); override;
    public
      constructor Create;
      destructor Destroy; override;
      function CurrentGazes : TNormalizedGazes; override;
  end;

implementation

uses
  opengaze, opengaze.helpers, sdl.app.video.methods, session.parameters.global;


const
  N = 30;

var
  NormalizedGazes : TNormalizedGazes;

{ TGazepointEyeTracker }

procedure TGazepointEyeTracker.SetGazeOnScreenEvent(AValue: TGazeOnScreenEvent);
begin
  if FReceiveGazeEvent = AValue then Exit;
  FReceiveGazeEvent := AValue;
end;

procedure TGazepointEyeTracker.SetOnCalibrationSuccessful(
  AValue: TNotifyEvent);
begin
  with OpenGazeControl do begin
    if Calibration.OnSuccess = AValue then Exit;
    Calibration.OnSuccess := AValue;
  end;
end;

procedure TGazepointEyeTracker.SetOnCalibrationFailed(
  AValue: TNotifyEvent);
begin
  with OpenGazeControl do begin
    if Calibration.OnFailed = AValue then Exit;
    Calibration.OnFailed := AValue;
  end;
end;

procedure TGazepointEyeTracker.StartRecording;
begin
  OpenGazeControl.StartRecording;
end;

procedure TGazepointEyeTracker.StopRecording;
begin
  OpenGazeControl.StopRecording;
end;

procedure TGazepointEyeTracker.StartCalibration;
begin
  OpenGazeControl.StartCalibration;
end;

procedure TGazepointEyeTracker.StopCalibration;
begin
  OpenGazeControl.StopCalibration;
end;

procedure TGazepointEyeTracker.CalibrationSuccessful;
begin
  // OpenGazeControl.StopCalibration;
end;

procedure TGazepointEyeTracker.SetDataFilename(AFilename: string);
begin
  OpenGazeControl.SetupDataFiles(AFilename);
end;

constructor TGazepointEyeTracker.Create;
var
  i : integer;
  BoundsRect : TSDL_Rect;
begin
  for i := Low(NormalizedGazes) to High(NormalizedGazes) do begin
    NormalizedGazes[i].X := 0.5;
    NormalizedGazes[i].Y := 0.5;
  end;

  BoundsRect := WindowBoundsRect;

  if GlobalTrialParameters.UseRemoteServer then begin
    OpenGazeControl.SetupForRemoteServer;
    OpenGazeControl.IP := '169.254.150.247';
  end;

  OpenGazeControl.Connect;
  if OpenGazeControl.Connected then begin
    OpenGazeControl.Recording.OnDataReceived := @ReceiveNormalizedGaze;
    OpenGazeControl.Calibration.Choreography.SetScreen(
      BoundsRect.x, BoundsRect.y, BoundsRect.w, BoundsRect.h);
  end;
end;

destructor TGazepointEyeTracker.Destroy;
begin

  inherited Destroy;
end;

function TGazepointEyeTracker.CurrentGazes: TNormalizedGazes;
begin
  Result := NormalizedGazes;
end;

procedure TGazepointEyeTracker.ReceiveNormalizedGaze(Sender : TObject;
  RawTag: TRawTag);
const
  BPOGX = 14; { if you want a dictionary }
  BPOGY = 15; { use TOpenGazeEvent instead of TGazeDataEvent}
  LLast = 29;
var
  event : TSDL_Event;
  i: Integer;
begin
  for i := Low(NormalizedGazes)+1 to High(NormalizedGazes) do begin
    NormalizedGazes[i-1] := NormalizedGazes[i];
  end;
  NormalizedGazes[LLast].X := RawTag.Pairs[BPOGX].Value.ToFloat;
  NormalizedGazes[LLast].Y := RawTag.Pairs[BPOGY].Value.ToFloat;

  event.type_ := EYE_TRACKER_GAZE_EVENT;
  event.user.data1 := Pointer(Self);
  SDL_PushEvent(@event);
end;

function TGazepointEyeTracker.IsFake: Boolean;
begin
  Result := OpenGazeControl.ProductID = 'NONE';
end;

function TGazepointEyeTracker.TrackerClassName: string;
begin
  Result := Self.ClassName;
end;

function TGazepointEyeTracker.GetGazeOnScreenEvent: TGazeOnScreenEvent;
begin
  Result := FReceiveGazeEvent;
end;

initialization
  NormalizedGazes := nil;
  SetLength(NormalizedGazes, N);

end.

