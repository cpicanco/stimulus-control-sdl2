{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.constants.trials;

{$mode ObjFPC}{$H+}

interface

type

  TTrialParameters = record
    Name : string;
    ReferenceName : string;
    Kind : string;
    Instruction : string;
    Cursor : integer;
    RepeatTrials : integer;
    LimitedHold : integer;
    InterTrialInterval : integer;
    TimeOutInterval : integer;
    HasConsequence : Boolean;
  end;

  TParserTrialsSourceKeys = record
    BlockIDKey : string;
    TrialIDKey : string;
    TrialIDSourceKey : string;
  end;

  TTrialKeys = record
    NameKey : string;
    ReferenceNameKey : string;
    CursorKey : string;
    KindKey : string;
    LimitedHoldKey : string;
    InterTrialIntervalKey : string;
    ConsequenceIntervalKey : string;
    RepeatTrialKey : string;
    TrialCountKey : string;
    HasConsequenceKey : string;
    InstructionKey : string;
    HasInstructionKey : string;
    HasCalibrationKey : string;
    IsTrue : string;
    IsFalse : string;
  end;

  TParserTrialsBase = record
    IDKey : string;
    KindKey : string;
    RepeatTrialKey : string;
    TrialCountKey : string;
    CursorKey : string;
    LimitedHoldKey : string;
    InterTrialIntervalKey : string;
    ConsequenceIntervalKey : string;
    HasConsequenceKey : string;
  end;

const
  HeaderName = 'Name';
  HeaderReferenceName = 'ReferenceName';
  HeaderKind = 'Kind';
  HeaderRepeatTrial = 'RepeatTrial';
  HeaderTrialCount = 'Trials';
  HeaderHasConsequence = 'HasConsequence';
  HeaderLimitedHold = 'LimitedHold';
  HeaderCursor = 'Cursor';
  HeaderInterTrialInterval = 'InterTrialInterval';
  HeaderConsequenceInterval = 'ConsequenceInterval';
  HeaderInstruction = 'Instruction';
  HeaderHasInstruction = 'HasInstruction';
  HeaderHasCalibration = 'DoCalibration';

  TrialKeys : TTrialKeys = (
    NameKey : HeaderName;
    ReferenceNameKey : HeaderName;
    CursorKey : HeaderCursor;
    KindKey : HeaderKind;
    LimitedHoldKey : HeaderLimitedHold;
    InterTrialIntervalKey : HeaderInterTrialInterval;
    ConsequenceIntervalKey : HeaderConsequenceInterval;
    RepeatTrialKey : HeaderRepeatTrial;
    TrialCountKey : HeaderTrialCount;
    HasConsequenceKey : HeaderHasConsequence;
    InstructionKey : HeaderInstruction;
    HasInstructionKey : HeaderHasInstruction;
    HasCalibrationKey : HeaderHasCalibration;
    IsTrue : 'T';
    IsFalse : 'F');

  ParserTrialsSourceKeys : TParserTrialsSourceKeys = (
    BlockIDKey : 'BlockID';
    TrialIDKey : 'TrialID';
    TrialIDSourceKey : 'TrialIDSource';
  );

  ParserTrialsBase : TParserTrialsBase = (
    IDKey : 'ID';
    KindKey : HeaderKind;
    RepeatTrialKey : HeaderRepeatTrial;
    TrialCountKey : HeaderTrialCount;
    CursorKey : HeaderCursor;
    LimitedHoldKey : HeaderLimitedHold;
    InterTrialIntervalKey : HeaderInterTrialInterval;
    ConsequenceIntervalKey : HeaderConsequenceInterval;
    HasConsequenceKey : HeaderHasConsequence;
  );

implementation

end.

