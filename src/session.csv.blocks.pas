{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.csv.blocks;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, session.csv.enumerable;

type

  { TCSVBlock }

  TCSVBlock = class(TCSVRows)
    private
      FMyName : string;
      FBlockID : Integer;
      FEndSessionOnCriterion : Boolean;
      FEndSessionOnNotCriterionAfterBlockRepetitions : Boolean;
      FRepeatStyle : string;
      FEndCriterionStyle : string;
      FEndCriterionEvaluationTime : string;
      FMaxBlockRepetitionConsecutives : integer;
      FMaxBlockRepetitionInSession	: integer;
      FNextBlockOnCriterion : integer;
      FNextBlockOnNotCriterion : integer;
      FEndCriterionValue : integer;
      FReinforcement : integer;
    protected
      procedure AfterLoadingParameters(Sender: TObject);
    public
      constructor Create; override;
      procedure AssignParameters(AParameters: TStringList); override;
      property MyName : string read FMyName write FMyName;
      property ID : integer read FBlockID write FBlockID;

      property NextBlockOnCriterion : Integer
        read FNextBlockOnCriterion
        write FNextBlockOnCriterion;

      property MaxBlockRepetitionConsecutives : Integer
        read FMaxBlockRepetitionConsecutives
        write FMaxBlockRepetitionConsecutives;

      property MaxBlockRepetitionInSession : Integer
        read FMaxBlockRepetitionInSession
        write FMaxBlockRepetitionInSession;

      property EndCriterionValue : Integer
        read FEndCriterionValue
        write FEndCriterionValue;

      property Reinforcement : Integer
        read FReinforcement
        write FReinforcement;

      property EndSessionOnCriterion : Boolean
        read FEndSessionOnCriterion
        write FEndSessionOnCriterion;

      property EndSessionOnNotCriterionAfterBlockRepetitions : Boolean
        read FEndSessionOnNotCriterionAfterBlockRepetitions
        write FEndSessionOnNotCriterionAfterBlockRepetitions;

      property RepeatStyle : string
        read FRepeatStyle
        write FRepeatStyle;

      property EndCriterionStyle : string
        read FEndCriterionStyle
        write FEndCriterionStyle;

      property EndCriterionEvaluationTime : string
        read FEndCriterionEvaluationTime
        write FEndCriterionEvaluationTime;

      property Values[const AKey: string]: string
        read GetValue write SetValue;
  end;

implementation

uses session.constants.blocks, session.configuration;

{ TCSVBlock }

procedure TCSVBlock.AfterLoadingParameters(Sender: TObject);
begin
  // base 0
  FBlockID -= 1;
  FNextBlockOnCriterion -= 1;
  FNextBlockOnNotCriterion -= 1;

  if FMyName.IsEmpty then begin
    FMyName := 'Block ' + (FBlockID+1).ToString;
  end;
end;

procedure TCSVBlock.AssignParameters(AParameters: TStringList);
begin
  inherited AssignParameters(AParameters);
end;

constructor TCSVBlock.Create;
var
  LRepeatStyle : TBlockRepeatStyle =
    TBlockRepeatStyle.None;

  LEndCriterionStyle : TBlockEndCriterionStyle =
    TBlockEndCriterionStyle.HitPorcentage;

  LBlockEndCriterionEvaluationTime : TBlockEndCriterionEvaluationTime =
    TBlockEndCriterionEvaluationTime.OnBlockEnd;
begin
  inherited Create;
  OnAfterLoadingParameters := @AfterLoadingParameters;
  FMyName := '';
  FBlockID := 0;
  FEndSessionOnCriterion := False;
  FEndSessionOnNotCriterionAfterBlockRepetitions := False;
  FRepeatStyle := LRepeatStyle.ToString;
  FEndCriterionStyle := LEndCriterionStyle.ToString;
  FEndCriterionEvaluationTime := LBlockEndCriterionEvaluationTime.ToString;
  FMaxBlockRepetitionConsecutives := 0;
  FMaxBlockRepetitionInSession	:= 0;
  FNextBlockOnCriterion := -1;
  FNextBlockOnNotCriterion := -1;
  FEndCriterionValue := 0;
  FReinforcement := 0;

  with ParserBlockKeys do begin
    RegisterParameter(
      IDKey,
      @FBlockID,
      FBlockID);

    RegisterParameter(
      NameKey,
      @FMyName,
      FMyName);

    RegisterParameter(
      EndSessionOnCriterionKey,
      @FEndSessionOnCriterion,
      FEndSessionOnCriterion);

    RegisterParameter(
      EndSessionOnNotCriterionAfterBlockRepetitionsKey,
      @FEndSessionOnNotCriterionAfterBlockRepetitions,
      FEndSessionOnNotCriterionAfterBlockRepetitions);

    RegisterParameter(
      RepeatStyleKey,
      @FRepeatStyle,
      FRepeatStyle);

    RegisterParameter(
      EndCriterionStyleKey,
      @FEndCriterionStyle,
      FEndCriterionStyle);

    RegisterParameter(
      EndCriterionEvaluationTimeKey,
      @FEndCriterionEvaluationTime,
      FEndCriterionEvaluationTime);

    RegisterParameter(
      MaxBlockRepetitionConsecutivesKey,
      @FMaxBlockRepetitionConsecutives,
      FMaxBlockRepetitionConsecutives);

    RegisterParameter(
      MaxBlockRepetitionInSessionKey,
      @FMaxBlockRepetitionInSession,
      FMaxBlockRepetitionInSession);

    RegisterParameter(
      NextBlockOnCriterionKey,
      @FNextBlockOnCriterion,
      FNextBlockOnCriterion);

    RegisterParameter(
      NextBlockOnNotCriterionKey,
      @FNextBlockOnNotCriterion,
      FNextBlockOnNotCriterion);

    RegisterParameter(
      EndCriterionValueKey,
      @FEndCriterionValue,
      FEndCriterionValue);

    RegisterParameter(
      ReinforcementKey,
      @FReinforcement,
      FReinforcement);
  end;
end;

end.

