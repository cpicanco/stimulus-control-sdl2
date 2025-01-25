
{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.constants.blocks;

{$mode ObjFPC}{$H+}

interface

type

  TParserBlockKeys = record
    IDKey : string;
    NameKey : string;
    EndSessionOnCriterionKey	: string;
    EndSessionOnNotCriterionAfterBlockRepetitionsKey : string;
    RepeatStyleKey : string;
    EndCriterionStyleKey : string;
    EndCriterionEvaluationTimeKey : string;
    MaxBlockRepetitionConsecutivesKey : string;
    MaxBlockRepetitionInSessionKey	: string;
    NextBlockOnCriterionKey : string;
    NextBlockOnNotCriterionKey : string;
    EndCriterionValueKey : string;
    ReinforcementKey : string;
  end;

const
  ParserBlockKeys : TParserBlockKeys = (
    IDKey : 'ID';
    NameKey : 'Name';
    EndSessionOnCriterionKey	: 'EndSessionOnCriterion';
    EndSessionOnNotCriterionAfterBlockRepetitionsKey : 'EndSessionOnNotCriterionAfterBlockRepetitions';
    RepeatStyleKey : 'RepeatStyle';
    EndCriterionStyleKey : 'EndCriterionStyle';
    EndCriterionEvaluationTimeKey : 'EndCriterionEvaluationTime';
    MaxBlockRepetitionConsecutivesKey : 'MaxBlockRepetitionConsecutives';
    MaxBlockRepetitionInSessionKey	: 'MaxBlockRepetitionInSession';
    NextBlockOnCriterionKey : 'NextBlockOnCriterion';
    NextBlockOnNotCriterionKey : 'NextBlockOnNotCriterion';
    EndCriterionValueKey : 'EndCriterionValue';
    ReinforcementKey: 'Reinforcement');

implementation

end.

