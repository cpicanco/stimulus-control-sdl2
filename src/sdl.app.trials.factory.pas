{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.trials.factory;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils, fgl
  , sdl.app.trials.contract
  , sdl.app.trials
  ;

type

  TTrialClass = class of TTrial;

  TTrialRegistry = specialize TFPGMap<string, TTrialClass>;

  { TTrialFactory }

  TTrialFactory = class sealed
    strict private
      class var CurrentTrial: TTrial;
      class var Registry: TTrialRegistry;
      class constructor Create;
      class destructor Destroy;
    public
      class procedure RegisterTrialClass(
        ATrialKind: string; ATrialClass: TTrialClass); static;
      class procedure Play; static;
      class function GetLastTrial : ITrial; static;
      class procedure Paint;
  end;

implementation

uses session.intertrial
   , session.configuration
   , session.pool
   , session.configurationfile
   , sdl.app.trials.mts
   , sdl.app.trials.dragdrop
   , sdl.app.trials.last
   ;

{ TTrialFactory }

class constructor TTrialFactory.Create;
begin
  Registry := TTrialRegistry.Create;
  CurrentTrial := nil;
end;

class destructor TTrialFactory.Destroy;
begin
  Registry.Free;
  if Assigned(CurrentTrial) then
    CurrentTrial.Free;
end;

class procedure TTrialFactory.RegisterTrialClass(ATrialKind: string;
  ATrialClass: TTrialClass);
begin
  Registry[ATrialKind] := ATrialClass;
end;

class procedure TTrialFactory.Play;
var
  TrialData : TTrialData;
  TrialClass : TTrialClass;
begin
  if Assigned(CurrentTrial) then
    FreeAndNil(CurrentTrial);

  TrialData := ConfigurationFile.CurrentTrial;

  if not Registry.TryGetData(TrialData.Kind, TrialClass) then
    raise EArgumentException.CreateFmt(
      'Trial kind is not registered: %s %s', [TrialData.Kind, TrialClass]);

  CurrentTrial := TrialClass.Create(nil);
  CurrentTrial.OnTrialEnd := InterTrial.OnBegin;
  CurrentTrial.Data := TrialData;
end;

class function TTrialFactory.GetLastTrial: ITrial;
var
  LMockData : TTrialData = (ID:-1; Kind : ''; Parameters:nil);
begin
  CurrentTrial := TLastTrial.Create(nil);
  CurrentTrial.OnTrialEnd := nil;
  CurrentTrial.Data := LMockData;
  Result := CurrentTrial as ITrial;
end;

class procedure TTrialFactory.Paint;
begin
  if Assigned(CurrentTrial) then
    CurrentTrial.AsIPaintable.Paint;
end;

initialization
  TTrialFactory.RegisterTrialClass(TMTS.ClassName, TMTS);
  TTrialFactory.RegisterTrialClass(TDragDrop.ClassName, TDragDrop);
  //TTrialFactory.RegisterTrialClass(TLastTrial.ClassName, TLastTrial);

end.
