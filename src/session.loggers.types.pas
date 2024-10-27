{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.loggers.types;

{$mode ObjFPC}{$H+}

interface

uses Math, SDL2, sdl.app.grids.types;

type
  { TDataProcedure }
  TDataProcedure = procedure (S : string) of object;

  // LGData have blc, trial data.
  // LGTimestamps for stm and response data.
  TLoggers = (LGData, LGTimestamps, LGInfo);

  TTimestampedEvent = record
    Timestamp : Float;
    Block : Word;
    Trial : Word;
    Code : string;
    Annotation : string;
  end;

  TInformation = record
    Version : UInt8;
    Basename : string;
    ParticipantName : string;
    SessionDesignFolder: string;
    SessionName : string;
    SessionResult : string;
    Grid : TMatrix;
    Monitor : TSDL_Rect;
    SessionStart : TDateTime;
    SessionEnd : TDateTime;
    SessionDuration : TDateTime;
  end;

implementation

end.

