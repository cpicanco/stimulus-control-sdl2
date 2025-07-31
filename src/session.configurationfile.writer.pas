{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.configurationfile.writer;

interface

uses
  Classes, SysUtils, Session.ConfigurationFile;

type

  { TConfigurationWriter }

  TConfigurationWriter = class
  private
    FBlockConfig: TStringList;
    FTrialConfig: TStringList;
    FTrialStartersConfig: TStringList;
    FCurrentBlock : integer;
    FConfigurationFile: TConfigurationFile;
    function GetCurrentTrial: integer;
    procedure SetStartTrial(AValue: TStartAt);
    procedure InvalidateTrials;
  public
    constructor Create(AConfigurationFile: TConfigurationFile); reintroduce;
    destructor Destroy; override;
    procedure Invalidate;
    procedure WriteBlock;
    procedure WriteTrial;
    procedure WriteStarter(ABlock, ATrial : integer);
    property BlockConfig : TStringList read FBlockConfig;
    property TrialConfig: TStringList read FTrialConfig;
    property TrialStartersConfig: TStringList read FTrialStartersConfig;
    property CurrentBlock  : integer read FCurrentBlock write FCurrentBlock;
    property CurrentTrial : integer read GetCurrentTrial;
    property StartAt : TStartAt write SetStartTrial;
  end;

implementation

uses session.configuration, session.trials.reinforcement;

{ TConfigurationWriter }

function TConfigurationWriter.GetCurrentTrial: integer;
begin
  Result := FConfigurationFile.Trials[CurrentBlock];
end;

procedure TConfigurationWriter.SetStartTrial(AValue: TStartAt);
begin
  FConfigurationFile.StartAt:=AValue;
end;

procedure TConfigurationWriter.InvalidateTrials;
var
  LBlockID, LTrialID : integer;
begin

end;

constructor TConfigurationWriter.Create(AConfigurationFile: TConfigurationFile);
begin
  if not Assigned(ConfigurationFile) then
    raise EFilerError.Create('Configuration file not assigned.');

  FConfigurationFile := AConfigurationFile;
  FBlockConfig := TStringList.Create;
  FTrialConfig := TStringList.Create;
  FTrialStartersConfig := TStringList.Create;
  FCurrentBlock := 0;
end;

destructor TConfigurationWriter.Destroy;
begin
  FTrialStartersConfig.Clear;
  FTrialStartersConfig.Free;
  FTrialConfig.Clear;
  FTrialConfig.Free;
  FBlockConfig.Clear;
  FBlockConfig.Free;
  FConfigurationFile := nil;
  inherited Destroy;
end;

procedure TConfigurationWriter.Invalidate;
var
  LTrial : integer;
  LBlock  : integer;
  LBlockData : TBlockConfiguration;
  LBlockReinforcement : TBlockReinforcement;
begin
  with FConfigurationFile do begin
    for LBlock := 0 to Blocks -1 do begin
      LBlockData := Block[LBlock];
      if LBlockData.Reinforcement < 100 then begin
        LBlockReinforcement :=
          BlockReinforcement(LBlockData.TotalTrials, LBlockdata.Reinforcement);
        for LTrial := 0 to Trials[LBlock] -1 do begin
          WriteToTrial(LTrial, LBlock,
            'HasConsequence', LBlockReinforcement[LTrial].ToString);
        end;
      end;
    end;
  end;
end;

procedure TConfigurationWriter.WriteBlock;
var
  i: integer;
  LName , LValue: string;
begin
  with FConfigurationFile do begin
    for i := 0 to FBlockConfig.Count -1 do begin
      FBlockConfig.GetNameValue(i, LName, LValue);
      WriteToBlock(Self.CurrentBlock, LName, LValue);
    end;
  end;
end;

procedure TConfigurationWriter.WriteTrial;
var
  i: integer;
  LName, LValue : string;
  LCurrentTrial : integer;
  LCurrentBlock  : integer;
begin
  LCurrentTrial := CurrentTrial;
  LCurrentBlock  := CurrentBlock;
  with FConfigurationFile do begin
    for i := 0 to FTrialConfig.Count -1 do begin
      FTrialConfig.GetNameValue(i, LName, LValue);
      WriteToTrial(LCurrentTrial, LCurrentBlock, LName, LValue);
    end;
  end;
end;

procedure TConfigurationWriter.WriteStarter(ABlock, ATrial: integer);
var
  LName, LValue : string;
  i: Integer;
begin
  with FConfigurationFile do begin
    for i := 0 to FTrialStartersConfig.Count -1 do begin
      FTrialStartersConfig.GetNameValue(i, LName, LValue);
      WriteToInstruction(ABlock, ATrial, LName, LValue);
    end;
  end;
end;

end.
