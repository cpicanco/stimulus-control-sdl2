{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.csv.trials;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils, session.csv.enumerable;

type

  { TCSVTrialsSource }

  TCSVTrialsSource = class(TCSVRows)
    private
      FBlockID  : Integer;
      FTrialID  : Integer;
      FTrialIDSource : string;
      procedure AfterLoadingParameters(Sender: TObject);
    public
      constructor Create; override;
      property TrialID : integer read FTrialID write FTrialID;
      property BlockID : integer read FBlockID write FBlockID;
      property TrialIDSource : string
        read FTrialIDSource write FTrialIDSource;
      property Values[const AKey: string]: string
        read GetValue write SetValue;
  end;

implementation

uses session.constants.trials;

{ TCSVTrialsSource }

procedure TCSVTrialsSource.AfterLoadingParameters(Sender: TObject);
begin
  FBlockID -= 1;
end;

constructor TCSVTrialsSource.Create;
begin
  inherited Create;
  OnAfterLoadingParameters := @AfterLoadingParameters;
  FTrialIDSource := '';
  FTrialID := 0;

  with ParserTrialsSourceKeys do begin
    RegisterParameter(BlockIDKey,
      @FBlockID, FBlockID);
    RegisterParameter(TrialIDKey,
      @FTrialID, FTrialID);
    RegisterParameter(TrialIDSourceKey,
      @FTrialIDSource, FTrialIDSource);
  end;
end;

end.


