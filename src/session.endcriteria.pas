{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.endcriteria;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils
  , session.configuration
  , session.configurationfile
  ;

type

  { TEndCriteria }

  TEndCriteria = class
  private
    FBlockID : integer;
    FTrialID : integer;
    FCurrentBlock : TBlockData;
    FCurrentTrial : TTrialData;
    function NextTrial : SmallInt;
    function NextBlock : SmallInt;
    function HitPorcentageInBlock : real;
    function IsEndBlock : Boolean;
    function IsEndSession : Boolean;
    function HitPorcentageCriterionAchieved : Boolean;
  public
    constructor Create;
    procedure InvalidateBlock;
    procedure InvalidateTrial(ATrialData : TTrialData);
    function OfSession : Boolean;
    function OfBlock : Boolean;
    function OfTrial : Boolean;
  end;

var
  EndCriteria : TEndCriteria;

implementation

uses
  session.pool
  , session.loggers.writerow
  ;

{ TEndCriteria }

constructor TEndCriteria.Create;
begin

end;

procedure TEndCriteria.InvalidateBlock;
begin
  FCurrentBlock := ConfigurationFile.CurrentBlock;
  ConfigurationFile.NewTrialOrder(FCurrentBlock);

  if FBlockID < ConfigurationFile.TotalBlocks then
    FBlockID := Pool.Block.ID;
  BlockName := FCurrentBlock.Name;
  FTrialID := 0;
end;

procedure TEndCriteria.InvalidateTrial(ATrialData : TTrialData);
begin
  FCurrentTrial := ATrialData;
  if FTrialID < FCurrentBlock.TotalTrials then begin
    FTrialID := Pool.Trial.ID;
  end;
  TrialName := FCurrentTrial.Parameters.Values['Name'];
end;

function TEndCriteria.OfSession: Boolean;
begin
  // TEndCriteria.OfSession is called every block end
  Result := IsEndSession;
end;

function TEndCriteria.OfBlock: Boolean;
begin
  // TEndCriteria.OfBlock is called every intertrial end
  // after TEndCriteria.OfTrial
  Result := IsEndBlock;
  if Result then begin
    Pool.Counters.EndBlock(NextBlock);
  end;
end;

function TEndCriteria.OfTrial: Boolean;
begin
  // TEndCriteria.OfTrial is called every intertrial end
  Pool.Counters.EndTrial(NextTrial);

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
    Result := FTrialID+1;
  end;

  if LRepeatValue > 0 then begin
    if Pool.Session.Block.Trial.Consecutives < LRepeatValue then begin
      Result := FTrialID;
    end;
  end;
  FTrialID := Result;
end;

function TEndCriteria.NextBlock: SmallInt;
var
  LRepeatStyle: TRepeatStyle;
  LRepeatValue: Integer;
begin
  LRepeatStyle := repsNone;
  if FCurrentBlock.MaxBlockRepetition > 0 then begin
    LRepeatValue := FCurrentBlock.MaxBlockRepetition -1;
    LRepeatStyle := repsConsecutive;
  end;

  if FCurrentBlock.MaxBlockRepetitionInSession > 0 then begin
    LRepeatValue := FCurrentBlock.MaxBlockRepetitionInSession -1;
    LRepeatStyle := repsGlobal;
  end;

  // go to next block by default
  Result := FBlockID+1;

  // go to back up block if it was setup and there are errors
  if (FCurrentBlock.NextBlockOnNotCriterion > -1) and
     (FCurrentBlock.BackUpBlockErrors > 0) then begin
    if Pool.Block.Events.Misses.Count >= FCurrentBlock.BackUpBlockErrors then begin

      // decide where to go base of repeat style
      case LRepeatStyle of

          // if none, just go to the block, may generate infinite loops
          repsNone : begin
            Result := FCurrentBlock.NextBlockOnNotCriterion;
          end;

          // if global, go to a different block
          repsGlobal: begin
            if LRepeatValue > 0 then begin
              if Pool.Session.Tree.Block[FBlockID].Count < LRepeatValue then begin
                Result := FCurrentBlock.NextBlockOnNotCriterion;
              end;
            end;
          end;

          // if consecutive, "go to" same block
          repsConsecutive: begin
            if Pool.Session.Block.Consecutives < LRepeatValue then begin
              Result := FBlockID;
            end;
          end;
        end;
      FBlockID := Result;
      Exit;
    end;
  end;


  if FCurrentBlock.CrtHitPorcentage > 0 then begin
    if HitPorcentageCriterionAchieved then begin
      if FCurrentBlock.NextBlockOnHitCriterion > -1 then begin
        Result := FCurrentBlock.NextBlockOnHitCriterion;
      end;

      if FCurrentBlock.EndSessionOnHitCriterion then begin
        Result := ConfigurationFile.TotalBlocks;
      end;
    end else begin
      //if FCurrentBlock.NextBlockOnNotCriterion > -1 then begin
      //  AGoToValue := FCurrentBlock.NextBlockOnNotCriterion;
      //end;
    end;
  end;
  FBlockID := Result;
end;

function TEndCriteria.IsEndSession: Boolean;

  procedure ForceEndSession;
  begin
    FBlockID := ConfigurationFile.TotalBlocks;
  end;

  procedure EvaluateCriteriaToForceEndSession;
  begin
    if FCurrentBlock.MaxBlockRepetitionInSession > 0 then begin
      if Pool.Session.Tree.Block[FBlockID].Count =
         FCurrentBlock.MaxBlockRepetitionInSession then begin
         ForceEndSession;
      end;
    end;
  end;

begin
  EvaluateCriteriaToForceEndSession;
  Result := FBlockID >= ConfigurationFile.TotalBlocks;
end;

function TEndCriteria.HitPorcentageCriterionAchieved: Boolean;
begin
  Result := HitPorcentageInBlock >= FCurrentBlock.CrtHitPorcentage;
end;

function TEndCriteria.HitPorcentageInBlock: real;
var
  LHits : integer;
begin
  LHits := Pool.Block.Events.Hits.Count;
  Result := (LHits * 100)/FCurrentBlock.TotalTrials;
end;

function TEndCriteria.IsEndBlock: Boolean;

  procedure ForceEndBlock;
  begin
    FTrialID := FCurrentBlock.TotalTrials;
  end;

  procedure EvaluateCriteriaToForceEndBlock;
  begin

  end;

begin
  EvaluateCriteriaToForceEndBlock;
  Result := FTrialID >= FCurrentBlock.TotalTrials;
end;


end.

