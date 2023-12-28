unit session.csv.trials.factory;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Generics.Collections, session.csv.trials.base;

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
  end;

implementation

uses
  session.csv.trials.pseudowords,
  session.csv.trials.pseudowords.cycle1,
  session.csv.trials.pseudowords.cycle2,
  session.csv.trials.pseudowords.cycle3,
  session.csv.trials.pseudowords.cycle4,
  session.csv.trials.pseudowords.cycle5,
  session.csv.trials.pseudowords.cycle6,
  session.csv.trials.dapaap;

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
  Result := CSVTrialClass.Create;
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

  //TCSVTrialsFactory.RegisterTrialClass('mm-eep', TDragDrop);
end.

