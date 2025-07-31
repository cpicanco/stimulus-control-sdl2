{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.endcriteria;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Math
  , session.configuration
  , session.configurationfile
  ;

type

  { TEndCriteria }

  TEndCriteria = class
  private
    FOnBlockCriterionAchieved : Boolean;
    FCurrentBlock : TBlockConfiguration;
    FCurrentTrial : TTrialConfiguration;
    FOnHitCriteriaAtSessionEnd: TNotifyEvent;
    FOnNotHitCriteriaAtSessionEnd: TNotifyEvent;
    function NextTrial : SmallInt;
    function NextBlock : SmallInt;
    function HitPorcentageInBlock : Float;
    function MissPorcentageInBlock : Float;
    function IsEndBlock(ATrialID : Word) : Boolean;
    function IsEndSession(ABlockID : Word) : Boolean;
    procedure SetOnHitCriteriaAtSessionEnd(AValue: TNotifyEvent);
    procedure SetOnNotHitCriteriaAtSessionEnd(AValue: TNotifyEvent);
    function ShouldEndBlock(var ATrialID : Word) : Boolean;
    function HitPercentageCriterionAchieved : Boolean;
    function MissPercentageCriterionAchieved : Boolean;
    function MissCountCriterionAchieved : Boolean;
    function HitCountCriterionAchieved : Boolean;
    function ConsecutiveHitsCriterionAchieved : Boolean;
    function ConsecutiveMissesCriterionAchieved : Boolean;
  public
    constructor Create;
    procedure InvalidateBlock;
    procedure InvalidateTrial(ATrialData : TTrialConfiguration);
    function OfSession : Boolean;
    function OfBlock : Boolean;
    function OfTrial : Boolean;
    property OnHitCriteriaAtSessionEnd : TNotifyEvent
      read FOnHitCriteriaAtSessionEnd write SetOnHitCriteriaAtSessionEnd;
    property OnNotHitCriteriaAtSessionEnd : TNotifyEvent
      read FOnNotHitCriteriaAtSessionEnd write SetOnNotHitCriteriaAtSessionEnd;
  end;

var
  EndCriteria : TEndCriteria;

implementation

uses
  session.pool
  , session.constants.trials
  , session.counters.all
  , session.loggers.writerow
  ;

{ TEndCriteria }

constructor TEndCriteria.Create;
begin
  inherited Create;

end;

procedure TEndCriteria.InvalidateBlock;
begin
  FCurrentBlock := ConfigurationFile.CurrentBlock;
  ConfigurationFile.NewOrdereringForTrialsInBlock(FCurrentBlock);
  BlockName := FCurrentBlock.Name;
  //Pool.Trial.ID := 0;
end;

procedure TEndCriteria.InvalidateTrial(ATrialData : TTrialConfiguration);
begin
  FCurrentTrial := ATrialData;
  TrialName := FCurrentTrial.Parameters.Values[TrialKeys.NameKey];
end;

function TEndCriteria.OfSession: Boolean;
begin
  // TEndCriteria.OfSession is called once every block end
  Result := IsEndSession(Pool.Block.ID);
end;

// TEndCriteria.OfBlock is called once every intertrial end
// after TEndCriteria.OfTrial
function TEndCriteria.OfBlock: Boolean;
var
  LNextBlock : SmallInt;
begin
  Result := ShouldEndBlock(Pool.Trial.ID);
  if Result then begin
    LNextBlock := NextBlock;
    Pool.Counters.BeforeEndBlock;
    Pool.Counters.EndBlock(LNextBlock);
  end;
end;

// TEndCriteria.OfTrial is called once every intertrial end
function TEndCriteria.OfTrial: Boolean;
var
  LNextTrial: Word;
begin
  LNextTrial := NextTrial;
  ShouldEndBlock(LNextTrial);
  Pool.Counters.BeforeEndTrial;
  Pool.Counters.EndTrial(LNextTrial);

  // if IsEndBlock reset NextTrial on next NextBlock call
  // this result does not have a function right now
  Result := True;
end;

function TEndCriteria.NextTrial: SmallInt;
var
  S1 : string = '';
  S2 : string = '';
  LRepeatValue: LongInt;
  LGoToTrial : SmallInt;
begin
  //LRepeatStyle := repsConsecutive;
  if Assigned(ConfigurationFile) then begin
    S1 := FCurrentTrial.Parameters.Values['RepeatTrial'];
    S2 := FCurrentTrial.Parameters.Values['GoToTrial'];   // TODO
  end;
  LRepeatValue := StrToIntDef(S1, 0) -1;
  LGoToTrial := StrToIntDef(S2, -1);

  if (LGoToTrial > -1) and (LGoToTrial < FCurrentBlock.TotalTrials) then begin
    Result := LGoToTrial;
  end else begin
    Result := Pool.Trial.ID+1;
  end;

  if LRepeatValue > 0 then begin
    if Pool.Session.Block.Trial.Consecutives < LRepeatValue then begin
      Result := Pool.Trial.ID;
    end;
  end;
end;

function TEndCriteria.NextBlock: SmallInt;
var
  LOnBlockCriterionAchieved : Boolean;

  function NextBlockOnCriterionAchieved : SmallInt;
  begin
    if FCurrentBlock.EndSessionOnCriterion then begin
      Result := ConfigurationFile.TotalBlocks;
    end else begin
      if FCurrentBlock.NextBlockOnCriterion > -1 then begin
        Result := FCurrentBlock.NextBlockOnCriterion;
      end else begin
        Result := Pool.Block.ID+1;
      end;
    end;
  end;

  function NextBlockAfterRepetitionDone(AValue : SmallInt): SmallInt;
  var
    LRepeatValue1: Integer;
    LRepeatValue2: Integer;

    procedure ForceEndSessionIfRequired;
    begin
      if FCurrentBlock.EndSessionOnNotCriterionAfterBlockRepetitions then begin
        Result := ConfigurationFile.TotalBlocks;
      end;
    end;

    procedure GoToNextBlock;
    begin
      Result := Pool.Block.ID+1;
    end;

    procedure GoToBlockIfValid;
    begin
      if AValue > -1 then begin
        Result := AValue;
      end;
    end;

    procedure RepeatBlock;
    begin
      Result := Pool.Block.ID;
    end;

  begin
    GoToNextBlock;
    case FCurrentBlock.RepeatStyle of
      None : begin
        GoToBlockIfValid;
        ForceEndSessionIfRequired;
      end;

      // repeat, then goto end session or go to NextBlockOnNotCriterion
      Consecutive: begin
        LRepeatValue1 := FCurrentBlock.MaxBlockRepetitionConsecutives -1;
        if LRepeatValue1 > 0 then begin
          LRepeatValue2 := Pool.Session.Block.Consecutives;
          if Pool.Session.Block.Consecutives < LRepeatValue1 then begin
            RepeatBlock;
          end else begin
            GoToBlockIfValid;
          end;
        end else begin
          ForceEndSessionIfRequired;
        end;
      end;

      // if below global repeat value
      // then go to the specified block
      // else end session if required
      Global: begin
        LRepeatValue1 := FCurrentBlock.MaxBlockRepetitionInSession -1;
        if LRepeatValue1 > 0 then begin
          if Pool.Session.Tree.Block[Pool.Block.ID].Count < LRepeatValue1 then begin
            GoToBlockIfValid;
          end else begin
            ForceEndSessionIfRequired;
          end;
        end else begin
          ForceEndSessionIfRequired;
        end;
      end;

      ConsecutiveAndGlobal: begin
        LRepeatValue1 := FCurrentBlock.MaxBlockRepetitionConsecutives -1;
        LRepeatValue2 := FCurrentBlock.MaxBlockRepetitionInSession -1;
        if (LRepeatValue1 > 0) and (LRepeatValue2 > 0) then begin
          if Pool.Session.Block.Consecutives < LRepeatValue1 then begin
            RepeatBlock;
          end else begin
            if Pool.Session.Tree.Block[Pool.Block.ID].Count < LRepeatValue2 then begin
              GoToBlockIfValid;
            end else begin
              ForceEndSessionIfRequired
            end;
          end;
        end else begin
          ForceEndSessionIfRequired;
        end;
      end;
    end;
  end;

  function NextBlockOnCriterionNotAchieved : SmallInt;
  begin
    if (FCurrentBlock.NextBlockOnNotCriterion > -1) or
       FCurrentBlock.EndSessionOnNotCriterionAfterBlockRepetitions then begin
      Result := NextBlockAfterRepetitionDone(
        FCurrentBlock.NextBlockOnNotCriterion);
    end else begin
      Result := Pool.Block.ID+1;
    end;
  end;

begin
  // go to next block by default
  Result := Pool.Block.ID+1;

  if FCurrentBlock.EndCriterionValue > 0 then begin
    case FCurrentBlock.EndCriterionEvaluationTime of
      OnTrialEnd : begin
        if FOnBlockCriterionAchieved then begin
          Result := NextBlockOnCriterionAchieved;
          if IsEndSession(Result) then begin
            OnHitCriteriaAtSessionEnd(Self);
          end;
        end else begin
          Result := NextBlockOnCriterionNotAchieved;
          if IsEndSession(Result) then begin
            OnNotHitCriteriaAtSessionEnd(Self);
          end;
        end;
      end;

      OnBlockEnd : begin
        case FCurrentBlock.EndCriterionStyle of
          HitCount:
            LOnBlockCriterionAchieved := HitCountCriterionAchieved;

          MissCount:
            LOnBlockCriterionAchieved := not MissCountCriterionAchieved;

          ConsecutiveHits:
            LOnBlockCriterionAchieved := ConsecutiveHitsCriterionAchieved;

          ConsecutiveMisses:
            LOnBlockCriterionAchieved := not ConsecutiveMissesCriterionAchieved;

          HitPorcentage:
            LOnBlockCriterionAchieved := HitPercentageCriterionAchieved;

          MissPorcentage:
            LOnBlockCriterionAchieved := not MissPercentageCriterionAchieved;
        end;

        if LOnBlockCriterionAchieved then begin
          Result := NextBlockOnCriterionAchieved;
          if IsEndSession(Result) then begin
            OnHitCriteriaAtSessionEnd(Self);
          end;
        end else begin
          Result := NextBlockOnCriterionNotAchieved;
          if IsEndSession(Result) then begin
            OnNotHitCriteriaAtSessionEnd(Self);
          end;
        end;
      end;
    end;
  end;
end;

function TEndCriteria.HitPercentageCriterionAchieved: Boolean;
begin
  Result := HitPorcentageInBlock >= FCurrentBlock.EndCriterionValue;
end;

function TEndCriteria.MissPercentageCriterionAchieved: Boolean;
begin
  Result := MissPorcentageInBlock >= FCurrentBlock.EndCriterionValue;
end;

function TEndCriteria.MissCountCriterionAchieved: Boolean;
begin
  Result := Pool.Block.Events.Misses.Count >= FCurrentBlock.EndCriterionValue;
end;

function TEndCriteria.HitCountCriterionAchieved: Boolean;
begin
  Result := Pool.Block.Events.Hits.Count >= FCurrentBlock.EndCriterionValue;
end;

function TEndCriteria.ConsecutiveHitsCriterionAchieved: Boolean;
begin
  Pool.Block.Events.Hits.FlushMaxConsecutives;
  Result := Pool.Block.Events.Hits.MaxConsecutives >= FCurrentBlock.EndCriterionValue;
end;

function TEndCriteria.ConsecutiveMissesCriterionAchieved: Boolean;
begin
  Pool.Block.Events.Misses.FlushMaxConsecutives;
  Result := Pool.Block.Events.Misses.MaxConsecutives >= FCurrentBlock.EndCriterionValue;
end;

function TEndCriteria.HitPorcentageInBlock: Float;
var
  LHits : integer;
begin
  LHits := Pool.Block.Events.Hits.Count;
  Result := (LHits * 100)/FCurrentBlock.TotalTrials;
end;

function TEndCriteria.MissPorcentageInBlock: Float;
var
  LMisses : integer;
begin
  LMisses := Pool.Block.Events.Misses.Count;
  Result := (LMisses * 100)/FCurrentBlock.TotalTrials;
end;

function TEndCriteria.IsEndBlock(ATrialID : Word): Boolean;
begin
  Result := ATrialID >= FCurrentBlock.TotalTrials;
end;

function TEndCriteria.IsEndSession(ABlockID: Word): Boolean;
begin
  Result := ABlockID >= ConfigurationFile.TotalBlocks;
end;

procedure TEndCriteria.SetOnHitCriteriaAtSessionEnd(AValue: TNotifyEvent);
begin
  if FOnHitCriteriaAtSessionEnd = AValue then Exit;
  FOnHitCriteriaAtSessionEnd := AValue;
end;

procedure TEndCriteria.SetOnNotHitCriteriaAtSessionEnd(AValue: TNotifyEvent);
begin
  if FOnNotHitCriteriaAtSessionEnd = AValue then Exit;
  FOnNotHitCriteriaAtSessionEnd := AValue;
end;

function TEndCriteria.ShouldEndBlock(var ATrialID: Word): Boolean;

  procedure ForceEndBlock;
  begin
    ATrialID := FCurrentBlock.TotalTrials;
  end;

  procedure EvaluateCriteriaToForceEndBlock;
  var
    LOnTrialCriterionAchieved: Boolean = False;
  begin
    if (FCurrentBlock.EndCriterionValue > 0) and
       (FCurrentBlock.EndCriterionEvaluationTime = OnTrialEnd) then begin

      case FCurrentBlock.EndCriterionStyle of
        HitCount:
          LOnTrialCriterionAchieved := HitCountCriterionAchieved;

        MissCount:
          LOnTrialCriterionAchieved := MissCountCriterionAchieved;

        ConsecutiveHits:
          LOnTrialCriterionAchieved := ConsecutiveHitsCriterionAchieved;

        ConsecutiveMisses:
          LOnTrialCriterionAchieved := ConsecutiveMissesCriterionAchieved;

        HitPorcentage:
          LOnTrialCriterionAchieved := HitPercentageCriterionAchieved;

        MissPorcentage:
          LOnTrialCriterionAchieved := MissPercentageCriterionAchieved;
      end;

      if LOnTrialCriterionAchieved then begin
        ForceEndBlock;
      end;

      case FCurrentBlock.EndCriterionStyle of
        HitCount, ConsecutiveHits, HitPorcentage: begin
          FOnBlockCriterionAchieved := LOnTrialCriterionAchieved;
        end;

        MissCount, ConsecutiveMisses, MissPorcentage: begin
          FOnBlockCriterionAchieved := not LOnTrialCriterionAchieved;
        end;
      end;
    end;
  end;

begin
  EvaluateCriteriaToForceEndBlock;
  Result := IsEndBlock(ATrialID);
end;


end.

