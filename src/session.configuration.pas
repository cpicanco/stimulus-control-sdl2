{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.configuration;

{$mode objfpc}{$H+}
{$modeswitch AdvancedRecords}

interface

uses Classes, SysUtils;

type
  TRepeatStyle = (repsNone, repsGlobal, repsConsecutive);

  { TTrialData }

  TTrialData = record
    ID : integer;
    Kind: string;
    ReferenceName: string;
    Parameters: TStringList;
    class operator = (A, B: TTrialData): Boolean;
  end;

  TTrials = array of TTrialData;

  { TBlockData }

  TBlockData = record
    ID : integer;
    Name: string;
    TotalTrials: integer;

    NextBlockOnNotCriterion : integer; // BackUpBlock	: integer;
    BackUpBlockErrors	: integer;
    MaxBlockRepetition : integer;
    MaxBlockRepetitionInSession	: integer;
    EndSessionOnHitCriterion	: Boolean;
    NextBlockOnHitCriterion : integer;
    CrtHitPorcentage : integer;

    Counter : string;
    AutoEndSession : Boolean;
    MaxCorrection: integer;
    BkGnd: integer;
    ITI: integer;
    DefNextBlock: string;
    CrtConsecutive: integer;
    CrtHitValue: integer;
    CrtConsecutiveHit: integer;
    CrtConsecutiveHitPerType : integer;
    CrtConsecutiveMiss : integer;
    CrtMaxTrials : integer;
    CrtCsqHit : integer;
    Trials: TTrials;
    class operator = (A, B: TBlockData): Boolean;
  end;

  TBlocks = array of TBlockData;

implementation


{ TTrialData }

class operator TTrialData.=(A, B: TTrialData): Boolean;
begin
  Result := A.ID = B.ID;
end;

{ TBlockData }

class operator TBlockData.=(A, B: TBlockData): Boolean;
begin
  Result := A.ID = B.ID;
end;

end.







