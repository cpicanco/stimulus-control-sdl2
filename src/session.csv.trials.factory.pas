{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.csv.trials.factory;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils, Generics.Collections, session.csv.trials.base;

type
  TCSVTrialClass = class of TCSVTrialsBase;

  TCSVTrialRegister = specialize TDictionary<string, TCSVTrialClass>;

  { TCSVTrialsFactory }

  TCSVTrialsFactory = class
    strict private
      class var Registry : TCSVTrialRegister;
    public
      class procedure RegisterCSVTrialClass(
        ATrialSourceFilename: string; ACSVTrialClass: TCSVTrialClass); static;
      class constructor Create;
      class destructor Destroy;
      class function New(ATrialSourceFilename : string) : TCSVTrialsBase;
      class procedure Initialize;
      class procedure Finalize;
  end;

implementation

uses
  picanco.experiments.words,
  session.csv.trials.pseudowords,
  session.csv.trials.pseudowords.cycle1,
  session.csv.trials.pseudowords.cycle2,
  session.csv.trials.pseudowords.cycle3,
  session.csv.trials.pseudowords.cycle4,
  session.csv.trials.pseudowords.cycle5,
  session.csv.trials.pseudowords.cycle6,
  session.csv.trials.dapaap,
  session.csv.trials.multisample;

{ TCSVTrialsFactory }

class procedure TCSVTrialsFactory.RegisterCSVTrialClass(
  ATrialSourceFilename: string; ACSVTrialClass: TCSVTrialClass);
begin
  Registry.Add(ATrialSourceFilename, ACSVTrialClass);
end;

class constructor TCSVTrialsFactory.Create;
begin
  Registry := TCSVTrialRegister.Create;
end;

class destructor TCSVTrialsFactory.Destroy;
begin
  Registry.Free;
end;

class function TCSVTrialsFactory.New(
  ATrialSourceFilename: string): TCSVTrialsBase;
var
  CSVTrialClass : TCSVTrialClass;
begin
  if not Registry.TryGetValue(ATrialSourceFilename, CSVTrialClass) then
    raise EArgumentException.CreateFmt(
      'CSV trial not registered: %s', [ATrialSourceFilename]);
  Result := CSVTrialClass.Create(ATrialSourceFilename);
end;

class procedure TCSVTrialsFactory.Initialize;
begin
  Picanco.Experiments.Words.Initialize;
end;

class procedure TCSVTrialsFactory.Finalize;
begin
  Picanco.Experiments.Words.Finalize;
end;

initialization
  TCSVTrialsFactory.RegisterCSVTrialClass(
    'mts-images', TCSVPseudowordsTrials);

  TCSVTrialsFactory.RegisterCSVTrialClass(
    'mts-pseudowords-1', TCSVPseudowordsCycle1);
  TCSVTrialsFactory.RegisterCSVTrialClass(
    'mts-pseudowords-2', TCSVPseudowordsCycle2);
  TCSVTrialsFactory.RegisterCSVTrialClass(
    'mts-pseudowords-3', TCSVPseudowordsCycle3);
  TCSVTrialsFactory.RegisterCSVTrialClass(
    'mts-pseudowords-4', TCSVPseudowordsCycle4);
  TCSVTrialsFactory.RegisterCSVTrialClass(
    'mts-pseudowords-5', TCSVPseudowordsCycle5);
  TCSVTrialsFactory.RegisterCSVTrialClass(
    'mts-pseudowords-6', TCSVPseudowordsCycle6);

  TCSVTrialsFactory.RegisterCSVTrialClass(
    'mts-dapa-ap', TCSVDAPAAPTrials);

  TCSVTrialsFactory.RegisterCSVTrialClass(
    'multi-sample', TCSVMultiSample);

  TCSVTrialsFactory.Initialize;

finalization
  TCSVTrialsFactory.Finalize;
end.

