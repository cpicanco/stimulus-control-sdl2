{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

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
  , session.configuration
  ;

type

  TTrialClass = class of TTrial;

  TTrialRegistry = specialize TDictionary<string, TTrialClass>;

  { TTrialFactory }

  TTrialFactory = class sealed
    strict private
      class var FCurrentTrial: TTrial;
      class var FRegistry: TTrialRegistry;
      class constructor Create;
      class destructor Destroy;
      class procedure InvalidateTrialData(
        out ATrialData : TTrialConfiguration; out ATrialClass : TTrialClass);
    public
      class procedure RegisterTrialClass(
        ATrialKind: string; ATrialClass: TTrialClass); static;
      class procedure Play; static;
      class procedure UpdateNavigator;
      class function CurrentTrial : ITrial;
      class function GetLastTrial : ITrial; static;
      class procedure FreeCurrentTrial;
      class function ToData : string; static;
  end;

implementation

uses Classes
   , session.intertrial

   , session.configurationfile
   , session.endcriteria
   , session.pool
   , sdl.app.controller.manager
   //, sdl.app.output
   , sdl.app.testmode
   , sdl.app.trials.mts
   , sdl.app.trials.dragdrop
   , sdl.app.trials.last
   ;

{ TTrialFactory }

class constructor TTrialFactory.Create;
begin
  FRegistry := TTrialRegistry.Create;
  FCurrentTrial := nil;
end;

class destructor TTrialFactory.Destroy;
begin
  FRegistry.Free;
end;

class procedure TTrialFactory.InvalidateTrialData(out
  ATrialData: TTrialConfiguration; out ATrialClass: TTrialClass);
begin
  ATrialData := ConfigurationFile.CurrentTrial;
  if not FRegistry.TryGetValue(ATrialData.Kind, ATrialClass) then
    raise EArgumentException.CreateFmt(
      'Trial kind is not registered: %s %s', [ATrialData.Kind, ATrialClass]);
  EndCriteria.InvalidateTrial(ATrialData);
end;

class procedure TTrialFactory.RegisterTrialClass(ATrialKind: string;
  ATrialClass: TTrialClass);
begin
  FRegistry.AddOrSetValue(ATrialKind, ATrialClass);
end;

class procedure TTrialFactory.Play;
var
  TrialData : TTrialConfiguration;
  TrialClass : TTrialClass;
begin
  FreeCurrentTrial;
  InvalidateTrialData(TrialData, TrialClass);

  FCurrentTrial := TrialClass.Create;
  FCurrentTrial.Navigator := Controllers.FirstController.Navigator;
  //FCurrentTrial.Parent := TSDLRenderer;
  FCurrentTrial.Name := 'T' + (Pool.Session.Trial.UID+1).ToString;
  FCurrentTrial.OnTrialEnd := InterTrial.OnBegin;
  FCurrentTrial.TestMode := TestMode;
  FCurrentTrial.Data := TrialData;
  FCurrentTrial.Show;

  Controllers.FirstController.Show;
end;

class procedure TTrialFactory.UpdateNavigator;
begin
  FCurrentTrial.Navigator := Controllers.FirstController.Navigator;
  FCurrentTrial.UpdateController;
end;

class function TTrialFactory.CurrentTrial: ITrial;
begin
  if FCurrentTrial <> nil then begin
    Result := FCurrentTrial as ITrial;
  end;
end;

class function TTrialFactory.GetLastTrial: ITrial;
var
  LMockData : TTrialConfiguration = (ID: -1 ; Kind : 'TLastTrial';
    ReferenceName: 'Mock'; Parameters: nil);
begin
  FreeCurrentTrial;
  Controllers.Disable;

  FCurrentTrial := TLastTrial.Create;
  FCurrentTrial.OnTrialEnd := nil;
  FCurrentTrial.Name := 'LastTrial';
  FCurrentTrial.Data := LMockData;
  Result := FCurrentTrial as ITrial;
end;

class procedure TTrialFactory.FreeCurrentTrial;
begin
  if Assigned(FCurrentTrial) then begin
    FCurrentTrial.Free;
    FCurrentTrial := nil;
  end;
end;

class function TTrialFactory.ToData: string;
begin
  if Assigned(FCurrentTrial) then begin
    Result := CurrentTrial.ToData;
  end;
end;

initialization
  TTrialFactory.RegisterTrialClass(TMTS.ClassName, TMTS);
  TTrialFactory.RegisterTrialClass(TDragDrop.ClassName, TDragDrop);
  //TTrialFactory.RegisterTrialClass(TLastTrial.ClassName, TLastTrial);

end.
