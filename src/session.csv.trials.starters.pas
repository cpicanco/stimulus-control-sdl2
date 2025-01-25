{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.csv.trials.starters;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils, session.csv.enumerable;

type

  { TCSVTrialsStarters }

  TCSVTrialsStarters = class(TCSVRows)
    private
      FBlockID        : integer;
      FTrialID        : integer;
      FInstruction    : string;
      FHasInstruction : boolean;
      FHasCalibration : boolean;
      procedure AfterLoadingParameters(Sender: TObject);
    public
      constructor Create; override;
      property BlockID  : integer read FBlockID;
      property TrialID : integer read FTrialID;
      property Values[const AKey: string]: string
        read GetValue write SetValue;
  end;

implementation

uses session.constants.trials.starters;

{ TCSVTrialsStarters }

procedure TCSVTrialsStarters.AfterLoadingParameters(Sender: TObject);
begin
  FBlockID -= 1;
  FTrialID -= 1;
  FHasInstruction := not FInstruction.IsEmpty;
end;

constructor TCSVTrialsStarters.Create;
begin
  inherited Create;
  OnAfterLoadingParameters := @AfterLoadingParameters;
  FBlockID        := 0;
  FTrialID        := 0;
  FInstruction    := '';
  FHasInstruction := False;
  FHasCalibration := False;

  with ParserTrialsStarters do begin
    RegisterParameter(BlockIDKey,
      @FBlockID, FBlockID);
    RegisterParameter(TrialIDKey,
      @FTrialID, FTrialID);
    RegisterParameter(InstructionKey,
      @FInstruction, FInstruction);
    RegisterParameter(HasInstructionKey,
      @FHasInstruction, FHasInstruction);
    RegisterParameter(HasCalibrationKey,
      @FHasCalibration, FHasCalibration);
  end;
end;

end.

