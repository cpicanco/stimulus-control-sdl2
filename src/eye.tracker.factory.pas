{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit eye.tracker.factory;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils, eye.tracker.types, eye.tracker.contract;

type

  { TEyeTrackerFactory }

  TEyeTrackerFactory = class
    class function New(ACode : TEyeTrackerCode) : IEyeTracker;
    class procedure Finalize;
  end;

implementation

uses
  eye.tracker.client,
  eye.tracker.pupil,
  eye.tracker.eyelink,
  eye.tracker.gazepoint;

var
  Client : TEyeTrackerClient = nil;

{ TEyeTrackerFactory }

class function TEyeTrackerFactory.New(ACode: TEyeTrackerCode): IEyeTracker;
begin
  case ACode of
    etGazepoint: Client := TGazepointEyeTracker.Create;
    etPupilLabs : Client := TPupilEyeTracker.Create;
    etEyeLink : Client := TEyeLinkEyeTracker.Create;

    otherwise begin
      Result := nil;
      Exit;
    end;
  end;
  Result := Client as IEyeTracker;
end;

class procedure TEyeTrackerFactory.Finalize;
begin
  Client.Free;
  Client := nil;
end;

end.

