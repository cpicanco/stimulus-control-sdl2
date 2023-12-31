{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit eye.tracker.eyelink;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , SDL2
  , eye.tracker.types
  , eye.tracker.client
  , eyelink.client
  , eyelink.data;

type

  { TEyeLinkEyeTracker }

  TEyeLinkEyeTracker = class sealed (TEyeTrackerClient)
    private
      FEyeLinkClient : TEyeLinkClient;
      FOnGazeOnScreenEvent : TGazeOnScreenEvent;
      procedure AllDataEvent(Sender: TObject; APALLF_DATA : array of PALLF_DATA);
    protected
      function GetGazeOnScreenEvent: TGazeOnScreenEvent; override;
      procedure SetGazeOnScreenEvent(
        AGazeOnScreenEvent: TGazeOnScreenEvent); override;
      procedure StartRecording; override;
      procedure StopRecording; override;
      procedure StartCalibration; override;
      procedure StopCalibration; override;
    public
      constructor Create;
      destructor Destroy; override;
  end;

implementation

uses session.pool;

{ TEyeLinkEyeTracker }

procedure TEyeLinkEyeTracker.SetGazeOnScreenEvent(
  AGazeOnScreenEvent: TGazeOnScreenEvent);
begin
  if FOnGazeOnScreenEvent = AGazeOnScreenEvent then Exit;
  FOnGazeOnScreenEvent := AGazeOnScreenEvent;
end;

procedure TEyeLinkEyeTracker.StartRecording;
const
  LRootFolder = '_eyelink_data' + DirectorySeparator;
begin
  FEyeLinkClient.OutputFolder := Pool.BaseFilename + LRootFolder;
  FEyeLinkClient.Start;
  //FEyeLinkClient.StartRealTime;
  FEyeLinkClient.StartDataRecording;
end;

procedure TEyeLinkEyeTracker.StopRecording;
begin
  //FEyeLinkClient.StopRealtime;
  FEyeLinkClient.StopDataRecording;
end;

procedure TEyeLinkEyeTracker.StartCalibration;
begin
  FEyeLinkClient.HostApp := Pool.App;
  FEyeLinkClient.DoTrackerSetup;
end;

procedure TEyeLinkEyeTracker.StopCalibration;
begin
  FEyeLinkClient.ExitCalibration;
end;

constructor TEyeLinkEyeTracker.Create;
begin
  FEyeLinkClient := TEyeLinkClient.Create;
  FEyeLinkClient.InitializeLibraryAndConnectToDevice;
end;

destructor TEyeLinkEyeTracker.Destroy;
begin
  FEyeLinkClient.Free;
  inherited Destroy;
end;

procedure TEyeLinkEyeTracker.AllDataEvent(Sender: TObject;
  APALLF_DATA: array of PALLF_DATA);
var
  LLastGazes : TGazes = nil;
  LLength : integer;
  i : integer;
  function NormToScreen(E : ALLF_DATA) : TGaze;
  begin
    if E.fe.eye = 2 then
      E.fe.eye := 0;
    Result.X :=
      Round(E.fs.gx[E.fe.eye]*FEyeLinkClient.HostApp.Monitor.w);
    Result.Y :=
      Round((1.0 - E.fs.gy[E.fe.eye])*FEyeLinkClient.HostApp.Monitor.h);
  end;

begin
  LLength := Length(APALLF_DATA);
  if LLength > 0 then begin
    SetLength(LLastGazes, LLength);
    for i := Low(LLastGazes) to High(LLastGazes) do begin
      LLastGazes[i] := NormToScreen(APALLF_DATA[i]^);
      if Assigned(FOnGazeOnScreenEvent) then begin
        FOnGazeOnScreenEvent(Self, LLastGazes);
      end;
    end;
  end;
end;

function TEyeLinkEyeTracker.GetGazeOnScreenEvent: TGazeOnScreenEvent;
begin
  Result := FOnGazeOnScreenEvent;
end;

end.
