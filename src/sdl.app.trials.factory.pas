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
  SysUtils, Generics.Collections
  , sdl.app.trials.contract
  , sdl.app.trials
  ;

type

  TTrialClass = class of TTrial;

  TTrialRegistry = specialize TDictionary<string, TTrialClass>;

  { TTrialFactory }

  TTrialFactory = class sealed
    strict private
      class var Registry: TTrialRegistry;
      class constructor Create;
      class destructor Destroy;
    public
      class var CurrentTrial: TTrial;
      class procedure RegisterTrialClass(
        ATrialKind: string; ATrialClass: TTrialClass); static;
      class procedure Play; static;
      class function GetLastTrial : ITrial; static;
      class procedure FreeCurrentTrial;
  end;

implementation

uses Classes
   , session.intertrial
   , session.configuration
   , session.configurationfile
   , session.endcriteria
   , session.pool
   , sdl.app.controller.manager
   , sdl.app.output
   , sdl.app.testmode
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
end;

class procedure TTrialFactory.RegisterTrialClass(ATrialKind: string;
  ATrialClass: TTrialClass);
begin
  Registry.AddOrSetValue(ATrialKind, ATrialClass);
end;

class procedure TTrialFactory.Play;
var
  TrialData : TTrialData;
  TrialClass : TTrialClass;
begin
  FreeCurrentTrial;

  TrialData := ConfigurationFile.CurrentTrial;
  if not Registry.TryGetValue(TrialData.Kind, TrialClass) then
    raise EArgumentException.CreateFmt(
      'Trial kind is not registered: %s %s', [TrialData.Kind, TrialClass]);
  EndCriteria.InvalidateTrial(TrialData);

  CurrentTrial := TrialClass.Create;
  CurrentTrial.Navigator := Controllers.FirstController.Navigator;
  //CurrentTrial.Parent := TSDLRenderer;
  CurrentTrial.Name := 'T' + (Pool.Session.Trial.UID+1).ToString;
  CurrentTrial.OnTrialEnd := InterTrial.OnBegin;
  CurrentTrial.TestMode := TestMode;
  CurrentTrial.Data := TrialData;
  CurrentTrial.Show;
end;

class function TTrialFactory.GetLastTrial: ITrial;
var
  LMockData : TTrialData = (ID: -1 ; Kind : 'TLastTrial';
    ReferenceName: 'Mock'; Parameters: nil);
begin
  FreeCurrentTrial;

  CurrentTrial := TLastTrial.Create;
  CurrentTrial.OnTrialEnd := nil;
  CurrentTrial.Name := 'LastTrial';
  CurrentTrial.Data := LMockData;
  CurrentTrial.Show;
  Result := CurrentTrial as ITrial;
end;

class procedure TTrialFactory.FreeCurrentTrial;
begin
  if Assigned(CurrentTrial) then begin
    CurrentTrial.Free;
    CurrentTrial := nil;
  end;
end;

initialization
  TTrialFactory.RegisterTrialClass(TMTS.ClassName, TMTS);
  TTrialFactory.RegisterTrialClass(TDragDrop.ClassName, TDragDrop);
  //TTrialFactory.RegisterTrialClass(TLastTrial.ClassName, TLastTrial);

end.
