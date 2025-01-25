{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit eye.tracker.fake;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils
  , SDL2
  , eye.tracker.types
  , eye.tracker.client;

type

  { TFakeEyeTracker }

  TFakeEyeTracker = class sealed (TEyeTrackerClient)
    private
      FMonitor : TSDL_Rect;
      FOnGazeOnScreenEvent : TGazeOnScreenEvent;
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
      procedure Invalidate;
  end;

implementation

uses sdl.app.video.methods;

{ TFakeEyeTracker }

procedure TFakeEyeTracker.SetGazeOnScreenEvent(
  AGazeOnScreenEvent: TGazeOnScreenEvent);
begin
  if FOnGazeOnScreenEvent = AGazeOnScreenEvent then Exit;
  FOnGazeOnScreenEvent := AGazeOnScreenEvent;
end;

procedure TFakeEyeTracker.StartRecording;
begin

end;

procedure TFakeEyeTracker.StopRecording;
begin

end;

procedure TFakeEyeTracker.StartCalibration;
begin

end;

procedure TFakeEyeTracker.StopCalibration;
begin

end;

constructor TFakeEyeTracker.Create;
begin

  Invalidate;
end;

destructor TFakeEyeTracker.Destroy;
begin

  inherited Destroy;
end;

procedure TFakeEyeTracker.Invalidate;
begin
  FMonitor := MonitorFromWindow;
end;

function TFakeEyeTracker.GetGazeOnScreenEvent: TGazeOnScreenEvent;
begin
  Result := FOnGazeOnScreenEvent;
end;

end.

