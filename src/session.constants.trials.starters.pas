{
  Stimulus Control
  Copyright (C) 2023-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.constants.trials.starters;

{$mode ObjFPC}{$H+}

interface

uses session.constants.trials;

type
  TParserTrialsStarters = record
    BlockIDKey        : string;
    TrialIDKey        : string;
    InstructionKey    : string;
    HasInstructionKey : string;
    HasCalibrationKey : string;
  end;

const
  ParserTrialsStarters : TParserTrialsStarters = (
    BlockIDKey        : 'Block';
    TrialIDKey        : 'Trial';
    InstructionKey    : HeaderInstruction;
    HasInstructionKey : HeaderHasInstruction;
    HasCalibrationKey : HeaderHasCalibration;
  );

implementation

end.

