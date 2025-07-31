{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.csv.trials.dapaap;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, session.csv.trials.mts;

type

  { TCSVDAPAAPTrials }

  TCSVDAPAAPTrials = class(TCSVTrialsMTS)
    private
      FName        : string;
      FRefName     : string;
      FSubset      : string;
      FSample      : string;
      FComparison1 : string;
      FComparison2 : string;
      FTotalLoops  : integer;
    protected
      procedure AfterLoadingParameters(Sender: TObject); override;
    public
      constructor Create(ASource: string); override;
      procedure AssignParameters(AParameters : TStringList); override;
      property Values[const AKey: string]: string
        read GetValue write SetValue;
  end;

implementation

uses
  session.constants.trials,
  session.constants.trials.dapaap;

{ TCSVDAPAAPTrials }

procedure TCSVDAPAAPTrials.AfterLoadingParameters(Sender: TObject);
begin
  inherited AfterLoadingParameters(Sender);
  if FComparison2.IsEmpty then begin
    Comparisons := 1;
  end else begin
    Comparisons := 2;
  end;
  FName := TrialID.ToString + #32 + '(' + FSample + #32 +
      Relation + #32 + FSubset + ')';
  FRefName := FSample;
end;

constructor TCSVDAPAAPTrials.Create(ASource: string);
begin
  inherited Create(ASource);
  OnAfterLoadingParameters := @AfterLoadingParameters;
  FName        := '';
  FRefName     := '';
  FSample      := '';
  FComparison1 := '';
  FComparison2 := '';
  FSubset      := '';
  FTotalLoops  := 0;

  with ParserTrialsDAPAAP do begin
    RegisterParameter(SubsetKey,
      @FSubset, FSubset);
    RegisterParameter(SampleKey+'1',
      @FSample, FSample);
    RegisterParameter(ComparisonKey+'1',
      @FSample, FSample);
    RegisterParameter(ComparisonKey+'2',
      @FComparison2, FComparison2);
    RegisterParameter(NameKey,
      @FName, FName);
    RegisterParameter(ReferenceNameKey,
      @FRefName, FRefName);
    RegisterParameter(TotalLoopsKey,
      @FTotalLoops, FTotalLoops);
  end;
end;

procedure TCSVDAPAAPTrials.AssignParameters(AParameters: TStringList);
begin
  inherited AssignParameters(AParameters);
end;

end.


